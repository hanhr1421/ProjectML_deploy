import os
import sys
import io
import json
import uuid
import time
import csv
import logging
import threading
import pika
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List
from contextlib import asynccontextmanager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("api_server")

# Config tu environment
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", 5672))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "guest")
QUEUE_NAME = os.getenv("QUEUE_NAME", "predict_queue")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/model")
TIMEOUT = int(os.getenv("RESPONSE_TIMEOUT", 120))


# Pydantic models cho request/response
class PredictRequest(BaseModel):
    product: str = Field(..., example="CO-A 1L X 12 BTL")
    bill_date: str = Field(..., example="2025-03-15")
    average_unit_price: float = Field(..., example=45000)
    note_promotion: int = Field(default=0)
    discount_promotion_code: int = Field(default=0)
    Lag_1: float = Field(default=0)
    Lag_7: float = Field(default=0)
    Lag_14: float = Field(default=0)
    Lag_30: float = Field(default=0)
    Lag_90: float = Field(default=0)
    Rolling_Mean_7: float = Field(default=0)
    Rolling_Mean_14: float = Field(default=0)
    Rolling_Mean_30: float = Field(default=0)
    Rolling_Std_7: float = Field(default=0)
    Rolling_Std_14: float = Field(default=0)
    Rolling_Std_30: float = Field(default=0)
    Expanding_Mean: float = Field(default=0)
    is_outlier: int = Field(default=0)


class PredictResponse(BaseModel):
    status: str
    product: str
    bill_date: str
    predicted_unit: float


class BatchRequest(BaseModel):
    requests: List[PredictRequest]


class BatchResponse(BaseModel):
    results: List[PredictResponse]
    total: int


class HealthResponse(BaseModel):
    status: str
    rabbitmq: str
    model_loaded: bool


# RabbitMQ RPC Client - gui message va cho ket qua
class RabbitMQClient:
    def __init__(self):
        self.connection = None
        self.channel = None
        self.callback_queue = None
        self.responses = {}
        self.lock = threading.Lock()

    def connect(self):
        credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
        params = pika.ConnectionParameters(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            credentials=credentials,
            heartbeat=600,
        )
        for attempt in range(1, 11):
            try:
                logger.info(f"Ket noi RabbitMQ ({attempt})...")
                self.connection = pika.BlockingConnection(params)
                self.channel = self.connection.channel()
                self.channel.queue_declare(queue=QUEUE_NAME, durable=True)

                # Tao callback queue de nhan ket qua tu consumer
                result = self.channel.queue_declare(queue="", exclusive=True)
                self.callback_queue = result.method.queue
                self.channel.basic_consume(
                    queue=self.callback_queue,
                    on_message_callback=self._on_response,
                    auto_ack=True,
                )
                logger.info("Da ket noi RabbitMQ!")
                return
            except Exception as e:
                if attempt == 10:
                    raise
                logger.warning(f"That bai: {e}. Thu lai sau 3s...")
                time.sleep(3)

    def _on_response(self, ch, method, properties, body):
        self.responses[properties.correlation_id] = json.loads(body.decode("utf-8"))

    def send_and_wait(self, request_data, timeout=TIMEOUT):
        """Gui message den consumer va cho ket qua tra ve"""
        with self.lock:
            corr_id = str(uuid.uuid4())
            self.responses[corr_id] = None

            self.channel.basic_publish(
                exchange="",
                routing_key=QUEUE_NAME,
                properties=pika.BasicProperties(
                    reply_to=self.callback_queue,
                    correlation_id=corr_id,
                    content_type="application/json",
                    delivery_mode=2,
                ),
                body=json.dumps(request_data),
            )

            start = time.time()
            while self.responses[corr_id] is None:
                self.connection.process_data_events(time_limit=1)
                if time.time() - start > timeout:
                    self.responses.pop(corr_id, None)
                    raise TimeoutError("Consumer khong phan hoi kip")

            return self.responses.pop(corr_id)

    def is_connected(self):
        return self.connection is not None and self.connection.is_open

    def close(self):
        if self.connection and self.connection.is_open:
            self.connection.close()


# Khoi tao
rpc_client = RabbitMQClient()
product_list = []


def load_product_list():
    sku_path = os.path.join(MODEL_DIR, "sku_mapping.json")
    if os.path.exists(sku_path):
        with open(sku_path) as f:
            return list(json.load(f).keys())
    return []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global product_list
    rpc_client.connect()
    product_list = load_product_list()
    logger.info(f"API da chay. {len(product_list)} san pham.")
    yield
    rpc_client.close()


app = FastAPI(
    title="Sales Forecasting API",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Routes ---

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_ui():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return HealthResponse(
        status="ok",
        rabbitmq="connected" if rpc_client.is_connected() else "disconnected",
        model_loaded=len(product_list) > 0,
    )


@app.get("/products", tags=["Products"])
def list_products():
    return {"total": len(product_list), "products": product_list}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_single(request: PredictRequest):
    try:
        result = rpc_client.send_and_wait({"type": "single", **request.model_dump()})
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message"))
        return PredictResponse(**result)
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Consumer timeout")


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(batch: BatchRequest):
    results = []
    for req in batch.requests:
        try:
            result = rpc_client.send_and_wait({"type": "single", **req.model_dump()})
            results.append(PredictResponse(**result))
        except Exception:
            results.append(PredictResponse(
                status="error", product=req.product,
                bill_date=req.bill_date, predicted_unit=-1,
            ))
    return BatchResponse(results=results, total=len(results))


@app.post("/predict/file", tags=["Prediction"])
async def predict_from_file(file: UploadFile = File(...)):
    """Upload file CSV -> consumer xu ly preprocessing -> forecast thang tiep theo"""
    content = await file.read()

    # Doc file CSV, xu ly encoding
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    reader = csv.DictReader(io.StringIO(text))
    raw_data = list(reader)

    if len(raw_data) == 0:
        raise HTTPException(status_code=400, detail="File CSV trong")

    # Lam sach ten cot (xu ly BOM)
    raw_data = [{k.strip().lstrip('\ufeff'): v for k, v in row.items()} for row in raw_data]

    # Kiem tra cot bat buoc
    required = {"bill_date", "product", "unit", "cost"}
    actual = set(raw_data[0].keys())
    missing = required - actual
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Thieu cot: {missing}. Co san: {sorted(actual)}"
        )

    logger.info(f"File '{file.filename}': {len(raw_data)} dong")

    # Chuyen kieu so cho cac cot
    for row in raw_data:
        for col in ["unit", "unit_price", "cost", "tax_rate"]:
            if col in row and row[col]:
                try:
                    row[col] = float(row[col])
                except (ValueError, TypeError):
                    row[col] = 0

    # Gui den consumer qua RabbitMQ
    try:
        result = rpc_client.send_and_wait(
            {"type": "raw_batch", "data": raw_data},
            timeout=TIMEOUT,
        )
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Consumer xu ly qua lau, vui long thu lai")

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
