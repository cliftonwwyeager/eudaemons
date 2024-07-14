import scapy.all as scapy
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import requests
import json
import logging
from scipy.stats import entropy, skew, kurtosis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

encoder = load_model('encoder_model.h5')
classifier = load_model('classifier_model.h5')
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

def ip_to_int(ip):
    """Convert IP string to integer."""
    return int.from_bytes(scapy.inet_aton(ip), 'big')

def capture_and_predict():
    while True:
        packets = scapy.sniff(count=100, timeout=30)
        features = extract_features(packets)
        scaled_features = scaler.transform(features)
        encoded_features = encoder.predict(scaled_features)
        predictions = classifier.predict(encoded_features)
        anomalies = [packet for packet, prediction in zip(packets, predictions) if prediction > 0.5]
        for anomaly in anomalies:
            ip = anomaly[scapy.IP].src
            block_ip(ip)

def extract_features(packets):
    features = []
    for packet in packets:
        packet_data = bytes(packet)
        feature_vector = [
            len(packet_data),
            np.mean(packet_data),
            np.std(packet_data),
            entropy(packet_data),
            skew(packet_data),
            kurtosis(packet_data),
        ]
        features.append(feature_vector)
    return np.array(features)

def block_ip(ip):
    url = "http://firewall-api/block"  # Adjust URL to your firewall's API
    data = {
        "ip": ip,
        "action": "block"
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        logging.info(f"Successfully blocked IP: {ip}")
    else:
        logging.error(f"Failed to block IP: {ip}")

if __name__ == "__main__":
    capture_and_predict()
