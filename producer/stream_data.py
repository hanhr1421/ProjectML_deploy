"""
Producer - doc file CSV va gui tung row vao RabbitMQ queue

Usage:
    python stream_data.py --mode setup --rabbitmq_server localhost
    python stream_data.py --mode teardown --rabbitmq_server localhost
"""

import argparse
import json
from datetime import datetime
from time import sleep

import pandas as pd
import pika

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m", "--mode",
    default="setup",
    choices=["setup", "teardown"],
    help="setup: gui data vao queue, teardown: xoa queue",
)
parser.add_argument(
    "-b", "--rabbitmq_server",
    default="localhost",
    help="RabbitMQ server address",
)
args = parser.parse_args()


def create_queue(channel, queue_name):
    """Tao queue neu chua co"""
    try:
        channel.queue_declare(queue=queue_name, durable=True)
        print(f"Queue '{queue_name}' created!")
    except Exception as e:
        print(f"Queue '{queue_name}' already exists. Error: {e}")


def create_streams(server, queue_name):
    """Ket noi RabbitMQ va gui data tu CSV vao queue"""
    connection = None
    channel = None

    # Thu ket noi 10 lan
    for i in range(10):
        try:
            crd = pika.credentials.PlainCredentials('guest', 'guest')
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=server, port=5672, credentials=crd)
            )
            channel = connection.channel()
            print("Connected to RabbitMQ!")
            break
        except Exception as e:
            print(f"Connection attempt {i+1} failed: {e}")
            sleep(10)

    # Doc CSV
    df = pd.read_csv('./sellout_w.csv')
    create_queue(channel, queue_name)

    print(f"Sending {len(df)} rows to queue '{queue_name}'...")

    # Gui tung row vao queue
    for idx, row in df.iterrows():
        record = row.to_dict()
        record["created"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=json.dumps(record, default=str),
            properties=pika.BasicProperties(delivery_mode=2),
        )

        if idx % 100 == 0:
            print(f"Sent {idx}/{len(df)} rows...")
        sleep(0.05)  # delay nho de khong qua tai

    print(f"Done! Sent {len(df)} rows to '{queue_name}'")


def teardown_queue(queue_name, server="localhost"):
    """Xoa queue"""
    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=server)
        )
        channel = connection.channel()
        channel.queue_delete(queue=queue_name)
        print(f"Queue '{queue_name}' deleted!")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parsed_args = vars(args)
    mode = parsed_args["mode"]
    server = parsed_args["rabbitmq_server"]
    queue_name = "forecast_queue"

    # Xoa queue cu truoc
    print("Cleaning up old queue...")
    try:
        teardown_queue(queue_name, server)
    except Exception:
        print("No existing queue to clean up")

    if mode == "setup":
        create_streams(server, queue_name)
