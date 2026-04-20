import os
import sys
import json
import time
import logging
import calendar
import numpy as np
import pandas as pd
import joblib
import pika
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("consumer")

# Doc config tu environment
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", 5672))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "guest")
QUEUE_NAME = os.getenv("QUEUE_NAME", "predict_queue")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/model")


def load_artifacts(model_dir):
    """Load model va cac file phu tro"""
    logger.info(f"Loading model tu {model_dir}...")
    model = joblib.load(os.path.join(model_dir, "rf_model.joblib"))
    le = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))

    with open(os.path.join(model_dir, "features.json")) as f:
        features = json.load(f)
    with open(os.path.join(model_dir, "sku_mapping.json")) as f:
        sku_mapping = json.load(f)
    with open(os.path.join(model_dir, "vietnam_holidays.json")) as f:
        holidays = pd.to_datetime(json.load(f))

    logger.info(f"Model: {type(model).__name__}, {len(features)} features, {len(sku_mapping)} san pham")
    return model, le, features, sku_mapping, holidays


def preprocess_raw_data(df, le, holidays):
    """Xu ly data tho tu CSV: aggregate, outlier, time features, lag, rolling, fourier"""
    df = df.copy()
    df['bill_date'] = pd.to_datetime(df['bill_date'], errors='coerce')
    df = df.dropna(subset=['bill_date'])

    # Bo cot khong can
    drop_cols = [c for c in ['customer_id', 'customer_name', 'address',
                              'product_code', 'product_name', 'unit_name',
                              'tax_rate', 'quantity_tons'] if c in df.columns]
    df = df.drop(columns=drop_cols, errors='ignore')

    if 'entity' in df.columns:
        df = df[df['entity'] != 'Gift']
        df = df.drop(columns=['entity'])

    # Chuyen cot so
    for col in ['unit', 'cost', 'unit_price']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Binary promotion
    if 'discount_promotion_code' in df.columns:
        df['discount_promotion_code'] = df['discount_promotion_code'].apply(lambda x: 0 if pd.isna(x) else 1)
    else:
        df['discount_promotion_code'] = 0

    if 'note_promotion' in df.columns:
        df['note_promotion'] = df['note_promotion'].apply(lambda x: 0 if pd.isna(x) else 1)
    else:
        df['note_promotion'] = 0

    # Aggregate theo ngay + san pham
    agg_df = df.groupby(['bill_date', 'product'], as_index=False).agg({
        'unit': 'sum',
        'note_promotion': 'max',
        'discount_promotion_code': 'max',
        'cost': 'sum',
    })

    # Tinh gia trung binh, tranh chia cho 0
    agg_df['average_unit_price'] = np.where(
        agg_df['unit'] != 0,
        agg_df['cost'] / agg_df['unit'],
        0
    )

    # Outlier detection bang z-score
    agg_df = agg_df.sort_values(['product', 'bill_date'])
    rm = agg_df.groupby('product')['unit'].transform(lambda x: x.rolling(30, min_periods=1).mean())
    rs = agg_df.groupby('product')['unit'].transform(lambda x: x.rolling(30, min_periods=1).std())
    z = np.where(rs > 0, (agg_df['unit'] - rm) / rs, 0)
    agg_df['is_outlier'] = (np.abs(z) > 3).astype(int)
    agg_df.drop(columns=['cost'], inplace=True)

    # Time features
    agg_df['day'] = agg_df['bill_date'].dt.day
    agg_df['month'] = agg_df['bill_date'].dt.month
    agg_df['year'] = agg_df['bill_date'].dt.year
    agg_df['day_of_week'] = agg_df['bill_date'].dt.dayofweek
    agg_df['week'] = agg_df['bill_date'].dt.isocalendar().week.astype(int)

    # Holiday week
    def check_holiday(d):
        return any((h - d).days in range(1, 8) for h in holidays)
    agg_df['holiday_week'] = agg_df['bill_date'].apply(check_holiday).astype(int)

    # Label encode product
    agg_df['product_encoded'] = agg_df['product'].apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )

    agg_df.set_index('bill_date', inplace=True)
    agg_df.sort_index(inplace=True)

    # Lag features
    for lag in [1, 7, 14, 30, 90]:
        agg_df[f'Lag_{lag}'] = agg_df.groupby('product_encoded')['unit'].shift(lag)

    # Rolling mean va std
    for w in [7, 14, 30]:
        agg_df[f'Rolling_Mean_{w}'] = agg_df.groupby('product_encoded')['unit'].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        agg_df[f'Rolling_Std_{w}'] = agg_df.groupby('product_encoded')['unit'].transform(
            lambda x: x.rolling(w, min_periods=1).std()
        )

    # Expanding mean
    agg_df['Expanding_Mean'] = agg_df.groupby('product_encoded')['unit'].transform(
        lambda x: x.expanding().mean()
    )

    # Fourier features
    agg_df['Fourier_Sin_7'] = np.sin(2 * np.pi * agg_df['day_of_week'] / 7)
    agg_df['Fourier_Cos_7'] = np.cos(2 * np.pi * agg_df['day_of_week'] / 7)
    agg_df['Fourier_Sin_30'] = np.sin(2 * np.pi * agg_df['day'] / 30)
    agg_df['Fourier_Cos_30'] = np.cos(2 * np.pi * agg_df['day'] / 30)

    # Fill NaN bang 0
    agg_df = agg_df.fillna(0)

    return agg_df


def forecast_next_month(agg_df, model, features, holidays):
    """Lay features cuoi cung moi san pham, tao row cho tung ngay thang ke tiep, predict roi sum"""

    last_date = agg_df.index.max()
    if last_date.month == 12:
        next_month, next_year = 1, last_date.year + 1
    else:
        next_month, next_year = last_date.month + 1, last_date.year

    days_in_month = calendar.monthrange(next_year, next_month)[1]
    forecast_dates = pd.date_range(
        start=f"{next_year}-{next_month:02d}-01",
        periods=days_in_month, freq='D'
    )

    logger.info(f"Du lieu: {agg_df.index.min().date()} -> {last_date.date()}")
    logger.info(f"Du bao: {next_year}-{next_month:02d} ({days_in_month} ngay)")

    # Lay tat ca san pham hop le
    products = agg_df[agg_df['product_encoded'] >= 0]['product'].unique()

    # Lay row cuoi cung lam base features
    last_rows = {}
    for prod in products:
        prod_data = agg_df[agg_df['product'] == prod]
        if len(prod_data) > 0:
            last_rows[prod] = prod_data.iloc[-1]

    logger.info(f"So san pham du bao: {len(last_rows)}")

    # Tinh holiday cho tung ngay
    holiday_flags = {}
    for date in forecast_dates:
        holiday_flags[date] = int(any((h - date).days in range(1, 8) for h in holidays))

    # Tao feature matrix
    future_rows = []
    row_meta = []

    for prod, base_row in last_rows.items():
        for date in forecast_dates:
            day = date.day
            month = date.month
            year = date.year
            dow = date.dayofweek
            week = date.isocalendar()[1]

            row = [
                base_row.get('product_encoded', 0),
                base_row.get('average_unit_price', 0),
                base_row.get('note_promotion', 0),
                base_row.get('discount_promotion_code', 0),
                base_row.get('Lag_1', 0),
                base_row.get('Lag_7', 0),
                base_row.get('Lag_14', 0),
                base_row.get('Lag_30', 0),
                base_row.get('Lag_90', 0),
                base_row.get('Rolling_Mean_7', 0),
                base_row.get('Rolling_Mean_14', 0),
                base_row.get('Rolling_Mean_30', 0),
                base_row.get('Rolling_Std_7', 0),
                base_row.get('Rolling_Std_14', 0),
                base_row.get('Rolling_Std_30', 0),
                base_row.get('Expanding_Mean', 0),
                np.sin(2 * np.pi * dow / 7),
                np.cos(2 * np.pi * dow / 7),
                np.sin(2 * np.pi * day / 30),
                np.cos(2 * np.pi * day / 30),
                day, month, year, dow, week,
                holiday_flags[date],
                0,  # is_outlier
            ]
            future_rows.append(row)
            row_meta.append(prod)

    # Predict tat ca cung luc
    X_future = np.nan_to_num(np.array(future_rows, dtype=float), nan=0.0)
    logger.info(f"Dang predict {X_future.shape[0]} rows...")
    all_preds = np.maximum(model.predict(X_future), 0)

    # Cong tong theo san pham
    product_totals = defaultdict(float)
    product_counts = defaultdict(int)
    for i, prod in enumerate(row_meta):
        product_totals[prod] += all_preds[i]
        product_counts[prod] += 1

    results = []
    for prod in product_totals:
        total = product_totals[prod]
        count = product_counts[prod]
        results.append({
            "product": prod,
            "forecast_month": f"{next_year}-{next_month:02d}",
            "predicted_total_unit": round(float(total), 2),
            "predicted_avg_daily": round(float(total / count), 2),
            "days_in_month": days_in_month,
            "status": "ok",
        })

    results.sort(key=lambda x: x['predicted_total_unit'], reverse=True)
    logger.info(f"Hoan tat: {len(results)} san pham")
    return results, f"{next_year}-{next_month:02d}"


def build_feature_vector(request, le, features, holidays):
    """Tao feature vector cho 1 request predict don le"""
    bill_date = pd.Timestamp(request["bill_date"])
    product_name = request["product"]
    product_encoded = le.transform([product_name])[0] if product_name in le.classes_ else 0

    day = bill_date.day
    month = bill_date.month
    year = bill_date.year
    dow = bill_date.dayofweek
    week = bill_date.isocalendar()[1]

    is_holiday = int(any((h - bill_date).days in range(1, 8) for h in holidays))

    fv = {
        "product_encoded": product_encoded,
        "average_unit_price": request.get("average_unit_price", 0),
        "note_promotion": request.get("note_promotion", 0),
        "discount_promotion_code": request.get("discount_promotion_code", 0),
        "Lag_1": request.get("Lag_1", 0),
        "Lag_7": request.get("Lag_7", 0),
        "Lag_14": request.get("Lag_14", 0),
        "Lag_30": request.get("Lag_30", 0),
        "Lag_90": request.get("Lag_90", 0),
        "Rolling_Mean_7": request.get("Rolling_Mean_7", 0),
        "Rolling_Mean_14": request.get("Rolling_Mean_14", 0),
        "Rolling_Mean_30": request.get("Rolling_Mean_30", 0),
        "Rolling_Std_7": request.get("Rolling_Std_7", 0),
        "Rolling_Std_14": request.get("Rolling_Std_14", 0),
        "Rolling_Std_30": request.get("Rolling_Std_30", 0),
        "Expanding_Mean": request.get("Expanding_Mean", 0),
        "Fourier_Sin_7": np.sin(2 * np.pi * dow / 7),
        "Fourier_Cos_7": np.cos(2 * np.pi * dow / 7),
        "Fourier_Sin_30": np.sin(2 * np.pi * day / 30),
        "Fourier_Cos_30": np.cos(2 * np.pi * day / 30),
        "day": day, "month": month, "year": year,
        "day_of_week": dow, "week": week,
        "holiday_week": is_holiday,
        "is_outlier": request.get("is_outlier", 0),
    }
    return np.array([[fv[f] for f in features]])


def on_message(channel, method, properties, body, model, le, features, holidays):
    """Xu ly message nhan tu RabbitMQ"""
    try:
        request = json.loads(body.decode("utf-8"))
        msg_type = request.get("type", "single")

        if msg_type == "raw_batch":
            raw_records = request["data"]
            logger.info(f"Nhan {len(raw_records)} dong du lieu")

            df = pd.DataFrame(raw_records)
            for col in ['unit', 'unit_price', 'cost', 'tax_rate', 'quantity_tons']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            agg_df = preprocess_raw_data(df, le, holidays)
            predictions, forecast_month = forecast_next_month(agg_df, model, features, holidays)

            # Tinh lich su theo thang cho bieu do
            hist_df = agg_df.copy()
            hist_df['year_month'] = hist_df['year'].astype(int).astype(str) + '-' + hist_df['month'].astype(int).astype(str).str.zfill(2)
            monthly_agg = hist_df.groupby(['product', 'year_month'])['unit'].sum().reset_index()

            monthly_history = {}
            for prod, grp in monthly_agg.groupby('product'):
                sorted_grp = grp.sort_values('year_month')
                monthly_history[prod] = {
                    "months": sorted_grp['year_month'].tolist(),
                    "units": [round(float(v), 2) for v in sorted_grp['unit'].values],
                }

            response = {
                "status": "ok",
                "forecast_month": forecast_month,
                "total_products": len(predictions),
                "total_raw_rows": len(raw_records),
                "results": predictions,
                "monthly_history": monthly_history,
            }
            logger.info(f"Du bao xong: {len(predictions)} san pham cho {forecast_month}")

        else:
            logger.info(f"Predict don: {request.get('product')}, {request.get('bill_date')}")
            X = build_feature_vector(request, le, features, holidays)
            prediction = max(0, float(model.predict(X)[0]))
            response = {
                "status": "ok",
                "product": request.get("product"),
                "bill_date": request.get("bill_date"),
                "predicted_unit": round(prediction, 2),
            }
            logger.info(f"Ket qua: {prediction:.2f} units")

    except Exception as e:
        logger.error(f"Loi xu ly: {e}", exc_info=True)
        response = {"status": "error", "message": str(e)}

    # Gui ket qua ve cho API
    if properties.reply_to:
        channel.basic_publish(
            exchange="",
            routing_key=properties.reply_to,
            properties=pika.BasicProperties(
                correlation_id=properties.correlation_id,
                content_type="application/json",
            ),
            body=json.dumps(response),
        )
    channel.basic_ack(delivery_tag=method.delivery_tag)


def connect_rabbitmq(max_retries=10, delay=5):
    """Ket noi RabbitMQ, thu lai neu that bai"""
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    params = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        credentials=credentials,
        heartbeat=600,
        blocked_connection_timeout=300,
    )
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Ket noi RabbitMQ ({attempt}/{max_retries})...")
            conn = pika.BlockingConnection(params)
            logger.info("Da ket noi RabbitMQ!")
            return conn
        except pika.exceptions.AMQPConnectionError as e:
            if attempt == max_retries:
                raise
            logger.warning(f"That bai: {e}. Thu lai sau {delay}s...")
            time.sleep(delay)


def main():
    model, le, features, sku_mapping, holidays = load_artifacts(MODEL_DIR)
    connection = connect_rabbitmq()
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    channel.basic_qos(prefetch_count=1)

    def callback(ch, method, properties, body):
        on_message(ch, method, properties, body, model, le, features, holidays)

    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback)
    logger.info(f"Dang cho message tren '{QUEUE_NAME}'...")

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        channel.stop_consuming()
    finally:
        connection.close()


if __name__ == "__main__":
    main()
