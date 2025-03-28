import os
import json
import pickle
import requests
import logging
import argparse
import threading
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.stats import median_abs_deviation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

app = Flask(__name__, template_folder='templates', static_folder='static')
ANOMALIES_LOG = []
MODEL_PERFORMANCE = []
EXPORTED_LOGS = []

FIREWALL_URL = os.environ.get("FIREWALL_URL", "https://fortigate.example.com")
SIEM_TYPE = os.environ.get("SIEM_TYPE", "elastic")
ELASTIC_URL = os.environ.get("ELASTIC_URL", "http://localhost:9200/firewall-logs/_doc")
CORTEX_URL = os.environ.get("CORTEX_URL", "http://cortex.example.com/api/logs")
SPLUNK_URL = os.environ.get("SPLUNK_URL", "http://splunk.example.com:8088/services/collector")
SENTINEL_URL = os.environ.get("SENTINEL_URL", "https://sentinel.example.com/api/logs")

def determine_firewall_vendor(url):
    lower_url = url.lower()
    if "forti" in lower_url:
        return "FortiGate"
    elif "palo" in lower_url or "pan" in lower_url:
        return "PaloAlto"
    elif "sonic" in lower_url:
        return "SonicWall"
    elif "meraki" in lower_url:
        return "Meraki"
    elif "unifi" in lower_url:
        return "Unifi"
    return "Unknown"

def get_log_location(vendor):
    if vendor == "FortiGate":
        return "/var/log/fortigate/traffic.log"
    elif vendor == "PaloAlto":
        return "/var/log/paloalto/traffic.log"
    elif vendor == "SonicWall":
        return "/var/log/sonicwall/traffic.log"
    elif vendor == "Meraki":
        return "/var/log/meraki/traffic.log"
    elif vendor == "Unifi":
        return "/var/log/ulog/syslogemu.log"
    return ""

def load_firewall_logs(path):
    if not os.path.exists(path):
        logging.error(f"Log file not found at {path}")
        return pd.DataFrame()
    with open(path, 'r') as f:
        data = []
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except Exception:
                    parts = line.split()
                    log_entry = {}
                    for part in parts:
                        if "=" in part:
                            key, value = part.split("=", 1)
                            log_entry[key.lower()] = value
                    if log_entry:
                        data.append(log_entry)
    return pd.DataFrame(data)

def logs_to_features(df):
    df = df.copy()
    if "application" in df.columns:
        le = LabelEncoder()
        df["application_enc"] = le.fit_transform(df["application"].astype(str))
    else:
        df["application_enc"] = 0

    def ip_to_int(ip):
        try:
            parts = ip.split(".")
            return (int(parts[0]) << 24) + (int(parts[1]) << 16) + (int(parts[2]) << 8) + int(parts[3])
        except Exception:
            return 0

    if "src" in df.columns:
        df["src_ip_int"] = df["src"].apply(ip_to_int)
    elif "src_ip" in df.columns:
        df["src_ip_int"] = df["src_ip"].apply(ip_to_int)
    else:
        df["src_ip_int"] = 0

    if "dst" in df.columns:
        df["dst_ip_int"] = df["dst"].apply(ip_to_int)
    elif "dst_ip" in df.columns:
        df["dst_ip_int"] = df["dst_ip"].apply(ip_to_int)
    else:
        df["dst_ip_int"] = 0

    cols = ["src_ip_int", "dst_ip_int", "src_port", "dst_port", "application_enc"]
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    data = df[cols].astype(np.float32)
    return data

def create_sequences(data, window_size=10):
    sequences = []
    for i in range(len(data) - window_size + 1):
        sequences.append(data[i:i+window_size])
    return np.array(sequences)

def detect_enhanced_anomalies(df, window_size=10):
    try:
        model = tf.keras.models.load_model("enhanced_autoencoder.h5")
    except Exception as e:
        logging.error(f"Cannot load 'enhanced_autoencoder.h5': {e}")
        return [], {}

    try:
        with open("scaler.pkl", "rb") as fh:
            scaler = pickle.load(fh)
    except Exception as e:
        logging.error(f"Cannot load 'scaler.pkl': {e}")
        return [], {}

    feats = logs_to_features(df)
    if feats.empty or len(feats) < window_size:
        logging.error("Not enough logs to create sequences for enhanced detection.")
        return [], {}

    scaled = scaler.transform(feats)
    sequences = create_sequences(scaled, window_size=window_size)
    reconstructions = model.predict(sequences)
    errors = np.mean(np.square(sequences - reconstructions), axis=(1, 2))
    median_err = np.median(errors)
    mad = median_abs_deviation(errors, scale="normal")
    threshold = median_err + 3.0 * mad
    anomaly_seq_indices = np.where(errors > threshold)[0]
    anomaly_indices = set()
    for i in anomaly_seq_indices:
        anomaly_indices.update(range(i, i + window_size))
    anomaly_indices = sorted(list(anomaly_indices))

    anomalies = []
    for idx in anomaly_indices:
        if idx < len(df):
            row = df.iloc[idx]
            info = f"src:{row.get('src','?')} dst:{row.get('dst','?')} app:{row.get('application','?')}"
            anomalies.append(info)
    
    performance_info = {
        "threshold": threshold,
        "median_error": median_err,
        "mad": mad
    }
    return anomalies, performance_info

def export_anomalies_to_siem(anomalies):
    exported = []
    now = datetime.utcnow().isoformat()
    if SIEM_TYPE.lower() == "elastic":
        for a in anomalies:
            doc = {"timestamp": now, "event": a}
            try:
                r = requests.post(ELASTIC_URL, json=doc)
                if r.status_code in [200, 201]:
                    exported.append(f"Exported to Elastic: {a}")
                else:
                    exported.append(f"Failed Elastic: {r.text}")
            except Exception as err:
                exported.append(f"Error Elastic: {err}")
    elif SIEM_TYPE.lower() == "cortex":
        for a in anomalies:
            doc = {"timestamp": now, "event": a}
            try:
                r = requests.post(CORTEX_URL, json=doc)
                if r.status_code in [200, 201]:
                    exported.append(f"Exported to Cortex: {a}")
                else:
                    exported.append(f"Failed Cortex: {r.text}")
            except Exception as err:
                exported.append(f"Error Cortex: {err}")
    elif SIEM_TYPE.lower() == "splunk":
        headers = {"Authorization": "Splunk <token>"}
        for a in anomalies:
            doc = {"time": now, "event": a}
            try:
                r = requests.post(SPLUNK_URL, json=doc, headers=headers)
                if r.status_code in [200, 201]:
                    exported.append(f"Exported to Splunk: {a}")
                else:
                    exported.append(f"Failed Splunk: {r.text}")
            except Exception as err:
                exported.append(f"Error Splunk: {err}")
    elif SIEM_TYPE.lower() == "sentinel":
        for a in anomalies:
            doc = {"timestamp": now, "event": a}
            try:
                r = requests.post(SENTINEL_URL, json=doc)
                if r.status_code in [200, 201]:
                    exported.append(f"Exported to Sentinel: {a}")
                else:
                    exported.append(f"Failed Sentinel: {r.text}")
            except Exception as err:
                exported.append(f"Error Sentinel: {err}")
    return exported

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html",
                           anomalies=ANOMALIES_LOG,
                           performance=MODEL_PERFORMANCE,
                           exported=EXPORTED_LOGS)

@app.route("/analyze", methods=["POST"])
def analyze_logs():
    ANOMALIES_LOG.clear()
    MODEL_PERFORMANCE.clear()
    vendor = determine_firewall_vendor(FIREWALL_URL)
    path = get_log_location(vendor)
    if not path:
        ANOMALIES_LOG.append("Unknown vendor or missing log path.")
        return render_template("index.html",
                               anomalies=ANOMALIES_LOG,
                               performance=MODEL_PERFORMANCE,
                               exported=EXPORTED_LOGS)
    df = load_firewall_logs(path)
    if df.empty:
        ANOMALIES_LOG.append("No logs found.")
        return render_template("index.html",
                               anomalies=ANOMALIES_LOG,
                               performance=MODEL_PERFORMANCE,
                               exported=EXPORTED_LOGS)
    anomalies, perf_info = detect_enhanced_anomalies(df, window_size=10)
    if anomalies:
        for a in anomalies:
            ANOMALIES_LOG.append(f"Anomaly: {a}")
    else:
        ANOMALIES_LOG.append("No anomalies detected.")
    if perf_info:
        MODEL_PERFORMANCE.append(f"Threshold: {perf_info['threshold']:.6f}")
        MODEL_PERFORMANCE.append(f"Median Error: {perf_info['median_error']:.6f}")
        MODEL_PERFORMANCE.append(f"MAD: {perf_info['mad']:.6f}")
    return render_template("index.html",
                           anomalies=ANOMALIES_LOG,
                           performance=MODEL_PERFORMANCE,
                           exported=EXPORTED_LOGS)

@app.route("/export", methods=["POST"])
def export_anomalies():
    EXPORTED_LOGS.clear()
    if not ANOMALIES_LOG:
        EXPORTED_LOGS.append("No anomalies to export.")
    else:
        anomalies = [x.replace("Anomaly: ", "") for x in ANOMALIES_LOG if x.startswith("Anomaly:")]
        if anomalies:
            result = export_anomalies_to_siem(anomalies)
            for r in result:
                EXPORTED_LOGS.append(r)
        else:
            EXPORTED_LOGS.append("No anomalies to export.")
    return render_template("index.html",
                           anomalies=ANOMALIES_LOG,
                           performance=MODEL_PERFORMANCE,
                           exported=EXPORTED_LOGS)

@app.route("/metrics", methods=["GET"])
def get_metrics():
    return jsonify({
        "anomalies": ANOMALIES_LOG,
        "performance": MODEL_PERFORMANCE,
        "exported": EXPORTED_LOGS
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=9500, type=int)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port)
    parser.add_argument("--port", default=9500, type=int)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port)
