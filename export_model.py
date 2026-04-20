# Train model va export artifacts
# Usage: python export_model.py --csv sellout_w.csv

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score


VIETNAM_HOLIDAYS = pd.to_datetime([
    "2023-01-01", "2023-01-20", "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24",
    "2023-04-30", "2023-05-01", "2023-09-02",
    "2024-01-01", "2024-02-08", "2024-02-09", "2024-02-10", "2024-02-11", "2024-02-12",
    "2024-04-30", "2024-05-01", "2024-09-02",
    "2025-01-01", "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31", "2025-02-01",
    "2025-04-30", "2025-05-01", "2025-09-02",
])

FEATURES = [
    'product_encoded', 'average_unit_price',
    'note_promotion', 'discount_promotion_code',
    'Lag_1', 'Lag_7', 'Lag_14', 'Lag_30', 'Lag_90',
    'Rolling_Mean_7', 'Rolling_Mean_14', 'Rolling_Mean_30',
    'Rolling_Std_7', 'Rolling_Std_14', 'Rolling_Std_30', 'Expanding_Mean',
    'Fourier_Sin_7', 'Fourier_Cos_7', 'Fourier_Sin_30', 'Fourier_Cos_30',
    'day', 'month', 'year', 'day_of_week', 'week', 'holiday_week', 'is_outlier'
]
TARGET = 'unit'


def is_holiday_week(date):
    return any((holiday - date).days in range(1, 8) for holiday in VIETNAM_HOLIDAYS)


def preprocess(csv_path):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    df['bill_date'] = pd.to_datetime(df['bill_date'])
    df = df.drop(columns=['customer_id', 'customer_name'])
    df = df.drop(df[df['entity'] == "Gift"].index)
    df = df.drop(columns=['entity'])

    df['discount_promotion_code'] = df['discount_promotion_code'].apply(lambda x: 0 if pd.isna(x) else 1)
    df['note_promotion'] = df['note_promotion'].apply(lambda x: 0 if pd.isna(x) else 1)

    agg_df = df.groupby(['bill_date', 'product'], as_index=False).agg({
        'unit': 'sum', 'note_promotion': 'max', 'discount_promotion_code': 'max', 'cost': 'sum'
    })
    agg_df['average_unit_price'] = agg_df['cost'] / agg_df['unit']

    # Outlier detection
    agg_df = agg_df.sort_values(['product', 'bill_date'])
    agg_df['rolling_mean'] = agg_df.groupby('product')['unit'].transform(lambda x: x.rolling(30).mean())
    agg_df['rolling_std'] = agg_df.groupby('product')['unit'].transform(lambda x: x.rolling(30).std())
    agg_df['z_score'] = (agg_df['unit'] - agg_df['rolling_mean']) / agg_df['rolling_std']
    agg_df['is_outlier'] = (agg_df['z_score'].abs() > 3).astype(int)

    # Time features
    agg_df['day'] = agg_df['bill_date'].dt.day
    agg_df['month'] = agg_df['bill_date'].dt.month
    agg_df['year'] = agg_df['bill_date'].dt.year
    agg_df['day_of_week'] = agg_df['bill_date'].dt.dayofweek
    agg_df['week'] = agg_df['bill_date'].dt.isocalendar().week.astype(int)
    agg_df['holiday_week'] = agg_df['bill_date'].apply(is_holiday_week).astype(int)

    agg_df = agg_df.drop(columns=['cost', 'rolling_mean', 'rolling_std', 'z_score'])

    # Label encode
    le = LabelEncoder()
    agg_df['product_encoded'] = le.fit_transform(agg_df['product'])

    agg_df.set_index('bill_date', inplace=True)
    agg_df.sort_index(inplace=True)

    # Lag features
    for lag in [1, 7, 14, 30, 90]:
        agg_df[f'Lag_{lag}'] = agg_df.groupby('product_encoded')['unit'].shift(lag)

    # Rolling
    for w in [7, 14, 30]:
        agg_df[f'Rolling_Mean_{w}'] = agg_df.groupby('product_encoded')['unit'].transform(lambda x: x.rolling(w).mean())
        agg_df[f'Rolling_Std_{w}'] = agg_df.groupby('product_encoded')['unit'].transform(lambda x: x.rolling(w).std())

    agg_df['Expanding_Mean'] = agg_df.groupby('product_encoded')['unit'].transform(lambda x: x.expanding().mean())

    # Fourier
    agg_df['Fourier_Sin_7'] = np.sin(2 * np.pi * agg_df['day_of_week'] / 7)
    agg_df['Fourier_Cos_7'] = np.cos(2 * np.pi * agg_df['day_of_week'] / 7)
    agg_df['Fourier_Sin_30'] = np.sin(2 * np.pi * agg_df['day'] / 30)
    agg_df['Fourier_Cos_30'] = np.cos(2 * np.pi * agg_df['day'] / 30)

    agg_df.drop(columns=['product'], inplace=True)
    print(f"Done: {agg_df.shape[0]} rows, {agg_df.shape[1]} columns")
    return agg_df, le


def train_and_export(csv_path, output_dir="model"):
    agg_df, le = preprocess(csv_path)

    # Split: last 2 months = test
    holdout_cutoff = agg_df.index.max() - pd.DateOffset(months=2)
    train_df = agg_df[agg_df.index <= holdout_cutoff].copy()
    holdout_df = agg_df[agg_df.index > holdout_cutoff].copy()
    print(f"Train: {train_df.shape[0]} | Test: {holdout_df.shape[0]}")

    # CV folds (time-based)
    unique_dates = train_df.index.unique().sort_values()
    n_dates = len(unique_dates)
    n_splits = 5
    date_folds = []
    for i in range(n_splits):
        train_end = int(n_dates * (i + 1) / (n_splits + 1))
        test_end = int(n_dates * (i + 2) / (n_splits + 1))
        train_mask = train_df.index.isin(unique_dates[:train_end])
        test_mask = train_df.index.isin(unique_dates[train_end:test_end])
        date_folds.append((np.where(train_mask)[0], np.where(test_mask)[0]))

    # GridSearch
    print("Running GridSearchCV...")
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid={'n_estimators': [100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]},
        scoring='neg_mean_absolute_error',
        cv=date_folds, n_jobs=-1, verbose=1,
    )
    grid.fit(train_df[FEATURES], train_df[TARGET])
    print(f"Best: {grid.best_params_}, CV MAE: {-grid.best_score_:.2f}")

    # Train final
    model = RandomForestRegressor(**grid.best_params_, random_state=42)
    model.fit(train_df[FEATURES], train_df[TARGET])

    # Evaluate
    preds = model.predict(holdout_df[FEATURES])
    print(f"Hold-out MAE: {mean_absolute_error(holdout_df[TARGET], preds):.2f}")
    print(f"Hold-out R2:  {r2_score(holdout_df[TARGET], preds):.4f}")

    # Export
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, "rf_model.joblib"))
    joblib.dump(le, os.path.join(output_dir, "label_encoder.joblib"))

    with open(os.path.join(output_dir, "features.json"), "w") as f:
        json.dump(FEATURES, f, indent=2)

    sku_mapping = {name: int(code) for name, code in zip(le.classes_, range(len(le.classes_)))}
    with open(os.path.join(output_dir, "sku_mapping.json"), "w") as f:
        json.dump(sku_mapping, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "best_params.json"), "w") as f:
        json.dump(grid.best_params_, f, indent=2)

    with open(os.path.join(output_dir, "vietnam_holidays.json"), "w") as f:
        json.dump([d.isoformat() for d in VIETNAM_HOLIDAYS], f, indent=2)

    print(f"Saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", default="model")
    args = parser.parse_args()
    train_and_export(args.csv, args.output)
