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
        return pd.DataFrame(scaler.transform(data), columns=['timestamp', 'proto', 'length', 'flags', 'window_size'])
    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
        return pd.DataFrame()

def detect_anomalies(cnn_model, bnn_model, scaler, data):
    if data.empty:
        return []
    try:
        processed_data = preprocess_data(data, scaler)
        cnn_predictions = cnn_model.predict(processed_data)
        bnn_predictions = bnn_model.predict(tf.constant(processed_data, dtype=tf.float32))
        combined_predictions = (cnn_predictions > 0.5) | (bnn_predictions > 0.5)
        return combined_predictions
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
        cnn_model = tf.keras.models.load_model('optimized_cnn_model.h5')
        bnn_model = tf.keras.models.load_model('optimized_bnn_model.h5')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        logging.error(f"Error loading models or scaler: {e}")
        return

    while True:
        packets = capture_packets()
        if packets:
            data = pd.DataFrame([{'timestamp': p.time, 'proto': p[scapy.IP].proto, 'length': len(p), 'flags': getattr(p[scapy.TCP], 'flags', 0), 'window_size': getattr(p[scapy.TCP], 'window', 0)} for p in packets if scapy.IP in p])
            if not data.empty:
                anomalies = detect_anomalies(cnn_model, bnn_model, scaler, data)
                anomaly_docs = data[anomalies].to_dict(orient='records')
                if anomaly_docs:
                    send_to_elasticsearch(anomaly_docs)

if __name__ == "__main__":
    main()