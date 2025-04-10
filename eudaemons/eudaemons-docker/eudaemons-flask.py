import os
import json
import logging
import argparse
import time
import redis
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template, jsonify, redirect, url_for, flash
from flask_socketio import SocketIO, emit
from flask_login import (LoginManager, UserMixin, login_user, logout_user,
                         current_user, login_required)
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import median_abs_deviation
from datetime import datetime

ANOMALIES_LOG = []
MODEL_PERFORMANCE = []
EXPORTED_LOGS = []
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "mysecretkey")
socketio = SocketIO(app, cors_allowed_origins="*")
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB   = int(os.environ.get("REDIS_DB", 0))
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
USERS = {
    "admin": {
        "id": "1",
        "username": "admin",
        "password_hash": generate_password_hash("adminpass"),
    }
}

class User(UserMixin):
    def __init__(self, user_id, username, password_hash):
        self.id = user_id
        self.username = username
        self.password_hash = password_hash

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    for _, data in USERS.items():
        if data["id"] == user_id:
            return User(data["id"], data["username"], data["password_hash"])
    return None

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uname = request.form.get("username")
        pwd   = request.form.get("password")
        for _, data in USERS.items():
            if data["username"] == uname:
                tmp_user = User(data["id"], data["username"], data["password_hash"])
                if tmp_user.verify_password(pwd):
                    login_user(tmp_user)
                    return redirect(url_for("index"))
        flash("Invalid credentials.")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

def load_model_and_scaler():
    model_path = "model_data/enhanced_autoencoder.h5"
    scaler_path = "model_data/scaler.pkl"
    threshold_path = "model_data/threshold.txt"
    if not os.path.exists(model_path):
        logging.error("No model found. Please train model first.")
        return None, None, None
    if not os.path.exists(scaler_path):
        logging.error("No scaler found. Please train model first.")
        return None, None, None
    if not os.path.exists(threshold_path):
        logging.error("No threshold file found. Please train model first.")
        return None, None, None
    model = tf.keras.models.load_model(model_path)
    with open(scaler_path, "rb") as fh:
        scaler = pickle.load(fh)
    with open(threshold_path, "r") as fth:
        threshold = float(fth.read().strip())
    return model, scaler, threshold

def logs_to_features(df):
    from sklearn.preprocessing import LabelEncoder
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
        except:
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

def detect_enhanced_anomalies(df, model, scaler, threshold, window_size=10):
    feats = logs_to_features(df)
    if feats.empty or len(feats) < window_size:
        return [], {}
    scaled = scaler.transform(feats)
    sequences = create_sequences(scaled, window_size=window_size)
    reconstructions = model.predict(sequences)
    errors = np.mean(np.square(sequences - reconstructions), axis=(1, 2))
    anomaly_seq_indices = np.where(errors > threshold)[0]
    anomaly_indices = set()
    for i in anomaly_seq_indices:
        anomaly_indices.update(range(i, i + window_size))
    anomaly_indices = sorted(list(anomaly_indices))
    anomalies = []
    for idx in anomaly_indices:
        if idx < len(df):
            row = df.iloc[idx]
            info = {
                "timestamp": row.get("timestamp", str(datetime.utcnow())),
                "src": row.get("src", row.get("src_ip","?")),
                "dst": row.get("dst", row.get("dst_ip","?")),
                "application": row.get("application","?"),
                "score": float(errors[min(i,len(errors)-1)]) / threshold  # example
            }
            anomalies.append(info)

    performance_info = {
        "threshold": threshold
    }
    return anomalies, performance_info

def export_anomalies_to_siem(anomalies):
    exported = []
    now = datetime.utcnow().isoformat()
    siem_type = os.environ.get("SIEM_TYPE", "elastic").lower()
    if siem_type == "elastic":
        elastic_url = os.environ.get("ELASTIC_URL", "http://localhost:9200/firewall-logs/_doc")
        for a in anomalies:
            doc = {"timestamp": now, "event": a}
            try:
                r = requests.post(elastic_url, json=doc)
                if r.status_code in [200, 201]:
                    exported.append("Exported to Elastic")
                else:
                    exported.append(f"Failed Elastic: {r.text}")
            except Exception as err:
                exported.append(f"Error Elastic: {err}")
    elif siem_type == "cortex":
        cortex_url = os.environ.get("CORTEX_URL", "http://cortex.example.com/api/logs")
        for a in anomalies:
            doc = {"timestamp": now, "event": a}
            try:
                r = requests.post(cortex_url, json=doc)
                if r.status_code in [200, 201]:
                    exported.append("Exported to Cortex")
                else:
                    exported.append(f"Failed Cortex: {r.text}")
            except Exception as err:
                exported.append(f"Error Cortex: {err}")
    elif siem_type == "splunk":
        splunk_url = os.environ.get("SPLUNK_URL", "http://splunk.example.com:8088/services/collector")
        headers = {"Authorization": "Splunk <token>"}
        for a in anomalies:
            doc = {"time": now, "event": a}
            try:
                r = requests.post(splunk_url, json=doc, headers=headers)
                if r.status_code in [200, 201]:
                    exported.append("Exported to Splunk")
                else:
                    exported.append(f"Failed Splunk: {r.text}")
            except Exception as err:
                exported.append(f"Error Splunk: {err}")
    elif siem_type == "sentinel":
        sentinel_url = os.environ.get("SENTINEL_URL", "https://sentinel.example.com/api/logs")
        for a in anomalies:
            doc = {"timestamp": now, "event": a}
            try:
                r = requests.post(sentinel_url, json=doc)
                if r.status_code in [200, 201]:
                    exported.append("Exported to Sentinel")
                else:
                    exported.append(f"Failed Sentinel: {r.text}")
            except Exception as err:
                exported.append(f"Error Sentinel: {err}")
    return exported

@app.route("/")
@login_required
def index():
    return render_template("index.html")
@app.route("/analyze", methods=["POST"])
@login_required
def analyze_logs():
    global ANOMALIES_LOG, MODEL_PERFORMANCE
    ANOMALIES_LOG.clear()
    MODEL_PERFORMANCE.clear()
    model, scaler, threshold = load_model_and_scaler()
    if not model:
        ANOMALIES_LOG.append("Model not found. Please train model first.")
        return redirect(url_for("index"))
    keys = list(r.scan_iter(match="firewall_log:*"))
    if not keys:
        ANOMALIES_LOG.append("No logs in Redis to analyze.")
        return redirect(url_for("index"))
    df_list = []
    for key in keys:
        entry = r.hgetall(key)
        decoded = {k.decode('utf-8'): v.decode('utf-8') for k, v in entry.items()}
        df_list.append(decoded)
    df_logs = pd.DataFrame(df_list)
    if df_logs.empty:
        ANOMALIES_LOG.append("No logs found in Redis.")
        return redirect(url_for("index"))
    if "timestamp" in df_logs.columns:
        df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"], errors='coerce')
        df_logs.sort_values("timestamp", inplace=True)
    anomalies, perf_info = detect_enhanced_anomalies(df_logs, model, scaler, threshold, window_size=10)
    if anomalies:
        for a in anomalies:
            ANOMALIES_LOG.append(f"Anomaly => src:{a['src']} dst:{a['dst']} app:{a['application']} score:{a['score']:.2f}")
    else:
        ANOMALIES_LOG.append("No anomalies detected.")
    if perf_info:
        MODEL_PERFORMANCE.append(f"Threshold: {perf_info['threshold']:.6f}")
    for anom in anomalies:
        socketio.emit("new_anomaly", anom, broadcast=True)

    return redirect(url_for("index"))

@app.route("/export", methods=["POST"])
@login_required
def export_anomalies():
    global EXPORTED_LOGS
    EXPORTED_LOGS.clear()
    anomalies = []
    for line in ANOMALIES_LOG:
        if line.startswith("Anomaly =>"):
            parts = line.replace("Anomaly => ", "").split()
            item = {"raw": line}
            anomalies.append(item)
    if not anomalies:
        EXPORTED_LOGS.append("No anomalies to export.")
        return redirect(url_for("index"))
    result = export_anomalies_to_siem(anomalies)
    for rmsg in result:
        EXPORTED_LOGS.append(rmsg)
    socketio.emit("export_update", {"exported": len(anomalies)}, broadcast=True)
    return redirect(url_for("index"))

@app.route("/metrics", methods=["GET"])
def get_metrics():
    return jsonify({
        "anomalies": ANOMALIES_LOG,
        "performance": MODEL_PERFORMANCE,
        "exported": EXPORTED_LOGS
    })

def metrics_background_thread():
    while True:
        socketio.sleep(2)
        total_logs = r.dbsize()
        total_anomalies = sum(1 for x in ANOMALIES_LOG if x.startswith("Anomaly =>"))
        data = {
            "total_logs": total_logs,
            "total_anomalies": total_anomalies,
            "ingest_rate": np.random.uniform(50, 150),
            "latency": np.random.uniform(5, 10),
            "elastic_exports": EXPORTED_LOGS.count("Exported to Elastic"),
            "splunk_exports": EXPORTED_LOGS.count("Exported to Splunk"),
            "cortex_exports": EXPORTED_LOGS.count("Exported to Cortex"),
            "sentinel_exports": EXPORTED_LOGS.count("Exported to Sentinel")
        }
        socketio.emit("metrics_update", data, broadcast=True)
@socketio.on('connect')

def handle_connect():
    logging.info(f"Client connected: {request.sid}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=9500, type=int)
    args = parser.parse_args()
    socketio.start_background_task(metrics_background_thread)
    socketio.run(app, host=args.host, port=args.port)
