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

ANOMALIES_LOG = []
LOGSTASH_STATUS = []
FIREWALL_URL = os.environ.get("FIREWALL_URL", "https://fortigate.example.com")
app = Flask(__name__)

HTML = """
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
</head>
<body>
  <h1>Firewall Log Anomaly Detection</h1>
  <form method="POST" action="/analyze">
    <button type="submit">Analyze Firewall Logs &amp; Detect Anomalies</button>
  </form>
  <form method="POST" action="/export">
    <button type="submit">Export Anomalies to Logstash</button>
  </form>
  <h2>Detected Anomalies</h2>
  <div class="scroll-box">
    {% for a in anomalies %}
      <div>{{ a }}</div>
    {% endfor %}
  </div>
  <h2>Logstash Export Status</h2>
  <div class="scroll-box">
    {% for s in statuses %}
      <div>{{ s }}</div>
    {% endfor %}
  </div>
</body>
</html>
"""

def determine_firewall_vendor(url: str) -> str:
    u = url.lower()
    if "forti" in u:
        return "FortiGate"
    elif "palo" in u or "pan" in u:
        return "PaloAlto"
    return "Unknown"

def get_log_location(vendor: str) -> str:
    if vendor == "FortiGate":
        return "/var/log/fortigate/traffic.log"
    elif vendor == "PaloAlto":
        return "/var/log/paloalto/traffic.log"
    return ""

def load_firewall_logs(path: str) -> pd.DataFrame:
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

def logs_to_features(df: pd.DataFrame) -> np.ndarray:
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
    sc = MinMaxScaler()
    return sc.fit_transform(data)

def detect_anomalies_from_logs(df: pd.DataFrame):
    try:
        model = tf.keras.models.load_model('trained_model.h5')
    except:
        logging.error("Cannot load 'trained_model.h5'.")
        return []
    try:
        with open('scaler.pkl','rb') as fh:
            scaler = pickle.load(fh)
    except:
        logging.error("Cannot load 'scaler.pkl'.")
        return []
    feats = logs_to_features(df)
    if len(feats) == 0:
        return []
    feats_scaled = scaler.transform(feats)
    preds = model.predict(feats_scaled)
    med = np.median(preds)
    mad = median_abs_deviation(preds, scale='normal')
    factor = 3.0
    threshold = med + factor * mad
    anom_idx = np.where(preds > threshold)[0]
    res = []
    for i in anom_idx:
        row = df.iloc[i]
        sinfo = f"src:{row.get('src_ip','?')} dst:{row.get('dst_ip','?')} app:{row.get('application','?')}"
        res.append(sinfo)
    return res

def export_to_logstash(anomalies):
    url = "http://localhost:5044"
    for e in anomalies:
        doc = {"timestamp": datetime.utcnow().isoformat(), "event": e}
        try:
            r = requests.post(url, json=doc)
            if r.status_code == 200:
                m = f"Sent: {doc}"
                LOGSTASH_STATUS.append(m)
                logging.info(m)
            else:
                m = f"Logstash error {r.status_code}"
                LOGSTASH_STATUS.append(m)
                logging.error(m)
        except Exception as err:
            m = f"Export error: {err}"
            LOGSTASH_STATUS.append(m)
            logging.error(m)

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML, anomalies=ANOMALIES_LOG, statuses=LOGSTASH_STATUS)

@app.route("/analyze", methods=["POST"])
def analyze_logs():
    v = determine_firewall_vendor(FIREWALL_URL)
    path = get_log_location(v)
    if not path:
        ANOMALIES_LOG.append("Unknown vendor or missing log path.")
        return render_template_string(HTML, anomalies=ANOMALIES_LOG, statuses=LOGSTASH_STATUS)
    df = load_firewall_logs(path)
    if df.empty:
        ANOMALIES_LOG.append("No logs found.")
        return render_template_string(HTML, anomalies=ANOMALIES_LOG, statuses=LOGSTASH_STATUS)
    res = detect_anomalies_from_logs(df)
    if res:
        for r in res:
            ANOMALIES_LOG.append(f"Anomaly: {r}")
    else:
        ANOMALIES_LOG.append("No anomalies detected.")
    return render_template_string(HTML, anomalies=ANOMALIES_LOG, statuses=LOGSTASH_STATUS)

@app.route("/export", methods=["POST"])
def export_anomalies():
    if not ANOMALIES_LOG:
        LOGSTASH_STATUS.append("No anomalies logged.")
    else:
        a = [x.replace("Anomaly: ","") for x in ANOMALIES_LOG if x.startswith("Anomaly:")]
        export_to_logstash(a)
    return render_template_string(HTML, anomalies=ANOMALIES_LOG, statuses=LOGSTASH_STATUS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=9500, type=int)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port)
