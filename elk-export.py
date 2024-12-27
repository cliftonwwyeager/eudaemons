import scapy.all as scapy
import requests
import logging
import pickle
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy, skew, kurtosis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def capture_packets(count=100, timeout=30):
    try:
        packets = scapy.sniff(count=count, timeout=timeout)
        return packets
    except Exception as e:
        logging.error(f"Failed to capture packets: {e}")
        return []

def export_to_logstash(events):
    # Adjust URL/port to match your Logstash HTTP input configuration
    logstash_url = "http://localhost:5044"
    for event in events:
        doc = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": str(event)
        }
        try:
            response = requests.post(logstash_url, json=doc)
            if response.status_code == 200:
                logging.info("Anomaly event sent to Logstash successfully.")
            else:
                logging.error(f"Logstash responded with status code {response.status_code}.")
        except Exception as e:
            logging.error(f"Failed to send event to Logstash: {e}")

def detect_anomalies(packets):
    model = tf.keras.models.load_model('trained_model.h5')
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    features = extract_features(packets)
    if len(features) == 0:
        return []
    features_scaled = scaler.transform(features)
    predictions = model.predict(features_scaled)
    anomalies = [pkt for pkt, pred in zip(packets, predictions) if pred > 0.5]
    return anomalies

def extract_features(packets):
    feature_list = []
    for packet in packets:
        try:
            raw_bytes = bytes(packet)
            feature_list.append([
                len(raw_bytes),
                entropy(raw_bytes) if len(raw_bytes) > 0 else 0,
                skew(raw_bytes) if len(raw_bytes) > 1 else 0,
                kurtosis(raw_bytes) if len(raw_bytes) > 1 else 0
            ])
        except Exception as e:
            logging.error(f"Error extracting features from a packet: {e}")
            feature_list.append([0, 0, 0, 0])
    return np.array(feature_list, dtype=np.float32)

if __name__ == "__main__":
    packets = capture_packets(count=100, timeout=30)
    if not packets:
        logging.info("No packets captured.")
    else:
        anomalies = detect_anomalies(packets)
        if anomalies:
            logging.info(f"Detected {len(anomalies)} anomaly/anomalies.")
            export_to_logstash(anomalies)
        else:
            logging.info("No anomalies detected.")
