import os
import json
import pickle
import requests
import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template_string
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.stats import median_abs_deviation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

app = Flask(__name__)
ANOMALIES_LOG = []
MODEL_PERFORMANCE = []
EXPORTED_LOGS = []
FIREWALL_URL = os.environ.get("FIREWALL_URL", "https://fortigate.example.com")
SIEM_TYPE = os.environ.get("SIEM_TYPE", "elastic")
ELASTIC_URL = os.environ.get("ELASTIC_URL", "http://localhost:9200/firewall-logs/_doc")
CORTEX_URL = os.environ.get("CORTEX_URL", "http://cortex.example.com/api/logs")
SPLUNK_URL = os.environ.get("SPLUNK_URL", "http://splunk.example.com:8088/services/collector")
SENTINEL_URL = os.environ.get("SENTINEL_URL", "https://sentinel.example.com/api/logs")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <style>
    body {
      background-color: #000;
      color: #0f0;
      font-family: monospace;
    }
    .scroll-box {
      height: 300px;
      overflow-y: scroll;
      border: 1px solid #0f0;
      margin-top: 10px;
      padding: 5px;
    }
  </style>
  <script src="scripts.js"></script>
</head>
<body>
  <h1>Firewall Log LSTM Anomaly Detection</h1>
  <form method="POST" action="/analyze">
    <button type="submit">Analyze Firewall Logs</button>
  </form>
  <form method="POST" action="/export">
    <button type="submit">Export Anomalies to SIEM</button>
  </form>
  <h2>Detected Anomalies</h2>
  <div class="scroll-box">
    {% for a in anomalies %}
      <div>{{ a }}</div>
    {% endfor %}
  </div>
  <h2>Model Performance</h2>
  <div class="scroll-box">
    {% for p in performance %}
      <div>{{ p }}</div>
    {% endfor %}
  </div>
  <h2>Exported Anomalies</h2>
  <div class="scroll-box">
    {% for e in exported %}
      <div>{{ e }}</div>
    {% endfor %}
  </div>
</body>
</html>
"""

def determine_firewall_vendor(url):
    lower_url = url.lower()
    if "forti" in lower_url:
        return "FortiGate"
    elif "palo" in lower_url or "pan" in lower_url:
        return "PaloAlto"
    return "Unknown"

def get_log_location(vendor):
    if vendor == "FortiGate":
        return "/var/log/fortigate/traffic.log"
    elif vendor == "PaloAlto":
        return "/var/log/paloalto/traffic.log"
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
                data.append(json.loads(line))
    return pd.DataFrame(data)

def logs_to_features(df):
    if "application" in df.columns:
        le = LabelEncoder()
        df["application_enc"] = le.fit_transform(df["application"].astype(str))
    else:
        df["application_enc"] = 0
    def ip_to_int(ip):
        try:
            parts = ip.split(".")
            return (int(parts[0]) << 24) + (int(parts[1]) << 16) + (int(parts[2]) << 8) + int(parts[3])
        except:
            return 0
    if "src_ip" in df.columns:
        df["src_ip_int"] = df["src_ip"].apply(ip_to_int)
    else:
        df["src_ip_int"] = 0
    if "dst_ip" in df.columns:
        df["dst_ip_int"] = df["dst_ip"].apply(ip_to_int)
    else:
        df["dst_ip_int"] = 0
    cols = ["src_ip_int","dst_ip_int","src_port","dst_port","application_enc"]
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    data = df[cols].astype(np.float32)
    return data

def reshape_for_lstm(X):
    return np.reshape(X, (X.shape[0], 1, X.shape[1]))

def detect_lstm_anomalies(df):
    try:
        model = tf.keras.models.load_model("lstm_autoencoder.h5")
    except:
        logging.error("Cannot load 'lstm_autoencoder.h5'.")
        return [], {}
    try:
        with open("scaler.pkl","rb") as fh:
            scaler = pickle.load(fh)
    except:
        logging.error("Cannot load 'scaler.pkl'.")
        return [], {}
    feats = logs_to_features(df)
    if feats.empty:
        return [], {}
    scaled = scaler.transform(feats)
    reshaped = reshape_for_lstm(scaled)
    recon = model.predict(reshaped)
    errors = np.mean(np.square(reshaped - recon), axis=(1,2))
    median_err = np.median(errors)
    mad = median_abs_deviation(errors, scale="normal")
    threshold = median_err + 3.0 * mad
    anomalies_idx = np.where(errors > threshold)[0]
    anomalies = []
    for i in anomalies_idx:
        row = df.iloc[i]
        info = f"src:{row.get('src_ip','?')} dst:{row.get('dst_ip','?')} app:{row.get('application','?')}"
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
    return render_template_string(
        HTML_TEMPLATE,
        anomalies=ANOMALIES_LOG,
        performance=MODEL_PERFORMANCE,
        exported=EXPORTED_LOGS
    )

@app.route("/analyze", methods=["POST"])
def analyze_logs():
    ANOMALIES_LOG.clear()
    MODEL_PERFORMANCE.clear()
    vendor = determine_firewall_vendor(FIREWALL_URL)
    path = get_log_location(vendor)
    if not path:
        ANOMALIES_LOG.append("Unknown vendor or missing log path.")
        return render_template_string(
            HTML_TEMPLATE,
            anomalies=ANOMALIES_LOG,
            performance=MODEL_PERFORMANCE,
            exported=EXPORTED_LOGS
        )
    df = load_firewall_logs(path)
    if df.empty:
        ANOMALIES_LOG.append("No logs found.")
        return render_template_string(
            HTML_TEMPLATE,
            anomalies=ANOMALIES_LOG,
            performance=MODEL_PERFORMANCE,
            exported=EXPORTED_LOGS
        )
    anomalies, perf_info = detect_lstm_anomalies(df)
    if anomalies:
        for a in anomalies:
            ANOMALIES_LOG.append(f"Anomaly: {a}")
    else:
        ANOMALIES_LOG.append("No anomalies detected.")
    if perf_info:
        MODEL_PERFORMANCE.append(f"Threshold: {perf_info['threshold']:.6f}")
        MODEL_PERFORMANCE.append(f"Median Error: {perf_info['median_error']:.6f}")
        MODEL_PERFORMANCE.append(f"MAD: {perf_info['mad']:.6f}")
    return render_template_string(
        HTML_TEMPLATE,
        anomalies=ANOMALIES_LOG,
        performance=MODEL_PERFORMANCE,
        exported=EXPORTED_LOGS
    )

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
    return render_template_string(
        HTML_TEMPLATE,
        anomalies=ANOMALIES_LOG,
        performance=MODEL_PERFORMANCE,
        exported=EXPORTED_LOGS
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=9500, type=int)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port)
