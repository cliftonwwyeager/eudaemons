import scapy.all as scapy
from elasticsearch import Elasticsearch
from datetime import datetime
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import logging
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def capture_packets():
    try:
        return scapy.sniff(count=100, timeout=30)
    except Exception as e:
        logging.error(f"Failed to capture packets: {e}")
        return []

def process_and_export_events(events):
    es = Elasticsearch(['http://localhost:9200'])  # Adjust the Elasticsearch server URL if necessary
    index_name = "network-anomalies"
    
    for event in events:
        doc = {
            'timestamp': datetime.utcnow(),
            'event': event
        }
        res = es.index(index=index_name, document=doc)
        logging.info(f"Indexed event to Elasticsearch: {res['result']}")

def detect_anomalies(packets):
    # Load the trained model
    model = tf.keras.models.load_model('trained_model.h5')  # Path to the trained model
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    features = extract_features(packets)
    scaled_features = scaler.transform(features)
    predictions = model.predict(scaled_features)
    anomalies = [packet for packet, prediction in zip(packets, predictions) if prediction > 0.5]
    return anomalies

def extract_features(packets):
    features = []
    for packet in packets:
        features.append([
            len(packet),
            entropy(packet),
            skew(packet),
            kurtosis(packet)
        ])
    return np.array(features)

if __name__ == "__main__":
    packets = capture_packets()
    anomalies = detect_anomalies(packets)
    if anomalies:
        process_and_export_events(anomalies)
    else:
        logging.info("No anomalies detected")