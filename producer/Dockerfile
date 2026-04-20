FROM python:3.11-slim

COPY producer/stream_data.py stream_data.py
COPY data/sellout_w.csv sellout_w.csv

RUN pip install --no-cache-dir pandas==2.2.3 pika==1.3.2

CMD ["python", "-u", "stream_data.py", "--mode", "setup", "--rabbitmq_server", "rabbitmq"]
