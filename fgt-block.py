import scapy.all as scapy
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import requests
import json

encoder = load_model('encoder_model.h5')
classifier = load_model('classifier_model.h5')
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

def ip_to_int(ip):
    """Convert IP string to integer."""
    return int.from_bytes(scapy.inet_aton(ip), 'big')

def capture_and_predict():
    while True:
        packets = scapy.sniff(count=1, filter="tcp", timeout=10)
        for packet in packets:
            if scapy.IP in packet and scapy.TCP in packet:
                data = {
                    "src": ip_to_int(packet[scapy.IP].src),
                    "dst": ip_to_int(packet[scapy.IP].dst),
                    "ttl": packet[scapy.IP].ttl,
                    "sport": packet[scapy.TCP].sport,
                    "dport": packet[scapy.TCP].dport
                }
                df = pd.DataFrame([data])
                transformed_data = scaler.transform(df[['src', 'dst', 'ttl', 'sport', 'dport']])
                df_transformed = pd.DataFrame(transformed_data, columns=['src', 'dst', 'ttl', 'sport', 'dport'])
                encoded_features = encoder.predict(df_transformed)
                prediction = classifier.predict(encoded_features)
                if prediction[0][0] > 0.75:
                    print(f"Anomaly detected from {packet[scapy.IP].src} to {packet[scapy.IP].dst} on port {packet[scapy.TCP].dport}")
                    block_host(packet[scapy.IP].src, packet[scapy.TCP].dport)

def block_host(ip_address, port):
    url = "http://fortigate.firewall/api/v2/cmdb/firewall/address"
    headers = {
        'Authorization': 'Bearer your_access_token_here',
        'Content-Type': 'application/json'
    }
    payload = {
        "name": f"block_{ip_address}_{port}",
        "subnet": f"{ip_address}/32",
        "associated-interface": "any",
        "comment": f"Blocked due to anomaly detection on port {port}"
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        print(f"Blocked IP address {ip_address} on port {port}")
    else:
        print(f"Failed to block IP address {ip_address} on port {port}: {response.text}")

if __name__ == "__main__":
    capture_and_predict()
