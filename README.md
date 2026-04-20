# Du bao doanh so - RabbitMQ + FastAPI

Deploy mo hinh du bao doanh so (RandomForest) su dung RabbitMQ va FastAPI.

## Yeu cau

- Docker
- Docker Compose
- Python 3.9+
## Cau truc project

```
ml-rabbitmq-deploy/
├── docker-compose.yaml
├── requirements-dev.txt
├── export_model.py
├── producer/
│   ├── Dockerfile
│   └── stream_data.py
├── consumer/
│   ├── Dockerfile
│   ├── consumer.py
│   └── requirements.txt
├── api/
│   ├── Dockerfile
│   ├── api_server.py
│   ├── requirements.txt
│   └── templates/
│       └── index.html
├── model/
│   ├── rf_model.joblib
│   ├── label_encoder.joblib
│   ├── features.json
│   ├── sku_mapping.json
│   └── vietnam_holidays.json
└── data/
    └── sellout_w.csv
```

## Kien truc

```
Producer (doc CSV) --> RabbitMQ (message queue) --> Consumer (predict) --> FastAPI (web UI)
```

- Producer: doc file CSV, gui tung dong vao RabbitMQ queue
- Consumer: nhan data, xu ly preprocessing, chay model predict
- FastAPI: web UI cho phep upload CSV va xem ket qua du bao
- RabbitMQ: message broker ket noi cac thanh phan

## Cach chay

### Buoc 1: Khoi dong

```bash
docker compose up -d --build
```

### Buoc 2: Su dung

Mo http://localhost:8000 tren trinh duyet:
- Keo tha file CSV (sellout_w.csv)
- Bam "Chay du bao"
- Xem ket qua va tai CSV|

## Thong tin model

- Thuat toan: RandomForestRegressor
- So luong features: 27
- So luong san pham: 132
- MAE (hold-out): 114.87
- R2 (hold-out): 0.7735

## Luu y

- File rf_model.joblib (~82MB) khong duoc push len GitHub do gioi han dung luong.
  Can tai rieng va dat vao thu muc model/ truoc khi chay Docker, link: [https://drive.google.com/file/d/1QNGdyVuUsbRAkI-Txvwo38Pbixn2_nAz/view?usp=drive_link]
