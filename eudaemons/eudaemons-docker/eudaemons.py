import os
import json
import argparse
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
import redis
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
try:
    import panos
    from panos.firewall import Firewall
    from panos.policies import SecurityRule
except ImportError:
    panos = None
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam

def fortigate_login(base_url, username, password):
    session = requests.Session()
    login_payload = {"username": username, "secretkey": password}
    url = f"{base_url}/logincheck"
    resp = session.post(url, data=login_payload, verify=False)
    if resp.status_code == 200 and "cookies" in resp.request.headers:
        csrftoken = session.cookies.get('ccsrftoken')
        token = csrftoken.strip('"') if csrftoken else ""
        return session, token
    else:
        raise ValueError("FortiGate login failed.")

def paloalto_login(base_url, username, password):
    params = {"type": "keygen", "user": username, "password": password}
    url = f"{base_url}/api/"
    resp = requests.get(url, params=params, verify=False)
    if resp.status_code == 200:
        try:
            from xml.etree import ElementTree
            root = ElementTree.fromstring(resp.text)
            key = root.find("./result/key").text
            return key
        except:
            raise ValueError("Failed to parse Palo Alto API key.")
    else:
        raise ValueError("Palo Alto login failed.")

def determine_firewall_vendor(firewall_url):
    fw_lower = firewall_url.lower()
    if "forti" in fw_lower:
        return "FortiGate"
    elif "palo" in fw_lower or "pan" in fw_lower:
        return "PaloAlto"
    else:
        return "Unknown"

def store_logs_in_redis(redis_host, redis_port, redis_db, log_data, redis_key="firewall_logs"):
    try:
        r = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        with r.pipeline() as pipe:
            for entry in log_data:
                pipe.rpush(redis_key, json.dumps(entry))
            pipe.execute()
    except Exception as e:
        print(f"[ERROR] Failed to store logs in Redis: {e}")

def retrieve_logs_from_redis(redis_host, redis_port, redis_db, redis_key="firewall_logs"):
    try:
        r = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        log_entries = r.lrange(redis_key, 0, -1)
        data = [json.loads(entry) for entry in log_entries]
        return pd.DataFrame(data)
    except Exception as e:
        print(f"[ERROR] Failed to retrieve logs from Redis: {e}")
        return pd.DataFrame()

def get_log_location(vendor):
    if vendor == "FortiGate":
        return "/var/log/fortigate/traffic.log"
    elif vendor == "PaloAlto":
        return "/var/log/paloalto/traffic.log"
    else:
        return ""

def load_firewall_logs(log_path):
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found at {log_path}")
    with open(log_path, 'r') as f:
        data = []
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return pd.DataFrame(data)

def filter_df_by_date_range(df, start_date, end_date, date_col="timestamp"):
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    return df[(df[date_col] >= start_date) & (df[date_col] <= end_date)].copy()

def preprocess_logs_for_model(df):
    if "application" in df.columns:
        app_encoder = LabelEncoder()
        df["application_enc"] = app_encoder.fit_transform(df["application"].astype(str))
    else:
        df["application_enc"] = 0
    def ip_to_int(ip_str):
        try:
            parts = ip_str.split(".")
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
    numeric_cols = ["src_ip_int", "dst_ip_int", "src_port", "dst_port", "application_enc"]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0
    data = df[numeric_cols].values.astype(np.float32)
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

def build_lstm_autoencoder(input_dim, latent_dim=16, dropout_rate=0.2, learning_rate=1e-3):
    model = Sequential()
    model.add(LSTM(32, activation='relu', return_sequences=True, input_shape=(1, input_dim)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(LSTM(latent_dim, activation='relu', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(TimeDistributed(Dense(input_dim)))
    model.compile(optimizer=Adam(learning_rate), loss='mse')
    return model

def train_autoencoder(model, X_train, X_val, epochs=10, batch_size=32):
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=1)
    return model

def calculate_dynamic_threshold(model, X_val, percentile=95):
    reconstructions_val = model.predict(X_val)
    mse_val = np.mean(np.square(X_val - reconstructions_val), axis=(1,2))
    return np.percentile(mse_val, percentile)

def detect_anomalies_autoencoder(model, data, threshold):
    reconstructions = model.predict(data)
    mse = np.mean(np.square(data - reconstructions), axis=(1,2))
    anomalies = mse > threshold
    return anomalies, mse

def get_anomaly_details(df, anomalies):
    anomaly_rows = df.iloc[anomalies.nonzero()[0]]
    src_ips = set(anomaly_rows["src_ip"].tolist()) if "src_ip" in df.columns else set()
    dst_ips = set(anomaly_rows["dst_ip"].tolist()) if "dst_ip" in df.columns else set()
    ports = set(anomaly_rows["dst_port"].tolist()) if "dst_port" in df.columns else set()
    apps = set(anomaly_rows["application"].tolist()) if "application" in df.columns else set()
    return {"src_ips": src_ips, "dst_ips": dst_ips, "ports": ports, "applications": apps}

def main_pipeline(firewall_url, baseline_start=None, baseline_end=None, forti_username="", forti_password="", palo_username="", palo_password="", use_redis=False, redis_host="localhost", redis_port=6379, redis_db=0):
    vendor = determine_firewall_vendor(firewall_url)
    if use_redis:
        df_logs = retrieve_logs_from_redis(redis_host, redis_port, redis_db, redis_key="firewall_logs")
        if df_logs.empty:
            return
    else:
        log_path = get_log_location(vendor)
        if not log_path:
            return
        df_logs = load_firewall_logs(log_path)
        store_logs_in_redis(redis_host, redis_port, redis_db, df_logs.to_dict(orient="records"))
    if baseline_start and baseline_end:
        try:
            s_date = pd.to_datetime(baseline_start)
            e_date = pd.to_datetime(baseline_end)
        except:
            return
        df_baseline = filter_df_by_date_range(df_logs, s_date, e_date)
        if len(df_baseline) == 0:
            return
        X_baseline = preprocess_logs_for_model(df_baseline)
        X_train, X_val = train_test_split(X_baseline, test_size=0.2, random_state=42)
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        model = build_lstm_autoencoder(X_baseline.shape[1], latent_dim=16, dropout_rate=0.2)
        model = train_autoencoder(model, X_train, X_val, epochs=5, batch_size=64)
        threshold = calculate_dynamic_threshold(model, X_val, percentile=95)
        df_outside_baseline = df_logs[~df_logs.index.isin(df_baseline.index)].copy()
        if len(df_outside_baseline) == 0:
            return
        X_outside = preprocess_logs_for_model(df_outside_baseline)
        X_outside = X_outside.reshape(X_outside.shape[0], 1, X_outside.shape[1])
        anomalies_bool, mse = detect_anomalies_autoencoder(model, X_outside, threshold)
        anomaly_details = get_anomaly_details(df_outside_baseline, anomalies_bool)
        print(anomaly_details)
    else:
        X_all = preprocess_logs_for_model(df_logs)
        X_train, X_val = train_test_split(X_all, test_size=0.2, random_state=42)
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        model = build_lstm_autoencoder(X_all.shape[1], latent_dim=16, dropout_rate=0.2)
        model = train_autoencoder(model, X_train, X_val, epochs=5, batch_size=64)
        threshold = calculate_dynamic_threshold(model, X_val, percentile=95)
        anomalies_bool, mse = detect_anomalies_autoencoder(model, X_all.reshape(X_all.shape[0], 1, X_all.shape[1]), threshold)
        anomaly_details = get_anomaly_details(df_logs, anomalies_bool)
        print(anomaly_details)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("-b", "--build-baseline", nargs=2, metavar=("START_DATE", "END_DATE"))
    parser.add_argument("--forti-username", default="")
    parser.add_argument("--forti-password", default="")
    parser.add_argument("--palo-username", default="")
    parser.add_argument("--palo-password", default="")
    parser.add_argument("--use-redis", action="store_true")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    args = parser.parse_args()
    if args.build_baseline:
        main_pipeline(args.url, baseline_start=args.build_baseline[0], baseline_end=args.build_baseline[1], forti_username=args.forti_username, forti_password=args.forti_password, palo_username=args.palo_username, palo_password=args.palo_password, use_redis=args.use_redis, redis_host=args.redis_host, redis_port=args.redis_port, redis_db=args.redis_db)
    else:
        main_pipeline(args.url, forti_username=args.forti_username, forti_password=args.forti_password, palo_username=args.palo_username, palo_password=args.palo_password, use_redis=args.use_redis, redis_host=args.redis_host, redis_port=args.redis_port, redis_db=args.redis_db)
