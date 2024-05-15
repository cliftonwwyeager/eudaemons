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

def preprocess_data(data, scaler):
    try:
        return pd.DataFrame(scaler.transform(data), columns=['src', 'dst', 'ttl'])
    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
        return pd.DataFrame()

def detect_anomalies(encoder, classifier, data):
    if data.empty:
        return []
    try:
        encoded_data = encoder.predict(data)
        predictions = classifier.predict(encoded_data)
        return predictions
    except Exception as e:
        logging.error(f"Error during anomaly detection: {e}")
        return []

def send_to_elasticsearch(docs):
    es = Elasticsearch("http://localhost:9200")
    index_name = 'network-anomalies'
    for doc in docs:
        doc['timestamp'] = datetime.now().isoformat()
        try:
            res = es.index(index=index_name, document=doc)
            logging.info(f"Document indexed, ID: {res['_id']}")
        except Exception as e:
            logging.error(f"Error sending document to Elasticsearch: {e}")

def main():
    try:
        encoder = tf.keras.models.load_model('encoder_model.h5')
        classifier = tf.keras.models.load_model('classifier_model.h5')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        logging.error(f"Error loading models or scaler: {e}")
        return

    while True:
        packets = capture_packets()
        if packets:
            data = pd.DataFrame([{'src': p[scapy.IP].src, 'dst': p[scapy.IP].dst, 'ttl': p[scapy.IP].ttl} for p in packets if scapy.IP in p])
            if not data.empty:
                processed_data = preprocess_data(data, scaler)
                anomalies = detect_anomalies(encoder, classifier, processed_data)
                anomaly_docs = data[anomalies > 0.5].to_dict(orient='records')
                if anomaly_docs:
                    send_to_elasticsearch(anomaly_docs)

if __name__ == "__main__":
    main()
