import os
import json
import uuid
import argparse
import logging
import time
import redis
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization,
                                     LSTM, RepeatVector, TimeDistributed,
                                     Conv1D, Bidirectional, Concatenate)
from tensorflow.keras.layers import Attention
from tensorflow.keras.optimizers import Adam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fortigate_login(base_url, username, password):
    session = requests.Session()
    login_payload = {"username": username, "secretkey": password}
    url = f"{base_url}/logincheck"
    resp = session.post(url, data=login_payload, verify=False)
    if resp.status_code == 200:
        csrftoken = session.cookies.get('ccsrftoken')
        token = csrftoken.strip('"') if csrftoken else ""
        logger.info("FortiGate login successful.")
        return session, token
    else:
        raise ValueError("FortiGate login failed.")

def fetch_fortigate_logs(session, token, base_url):
    url = f"{base_url}/api/logs"
    headers = {"X-CSRF-Token": token} if token else {}
    resp = session.post(url, data={}, verify=False)
    if resp.status_code == 200:
        try:
            data = resp.json()
            logger.info("Successfully fetched FortiGate logs via API.")
            return pd.DataFrame(data)
        except Exception as e:
            raise ValueError(f"Failed to parse FortiGate logs from API: {e}")
    else:
        raise ValueError("Failed to fetch FortiGate logs from API.")

def paloalto_login(base_url, username, password):
    params = {"type": "keygen", "user": username, "password": password}
    url = f"{base_url}/api/"
    resp = requests.get(url, params=params, verify=False)
    if resp.status_code == 200:
        from xml.etree import ElementTree
        root = ElementTree.fromstring(resp.text)
        key = root.find("./result/key").text
        logger.info("Palo Alto login successful.")
        return key
    else:
        raise ValueError("Palo Alto login failed.")

def fetch_paloalto_logs(api_key, base_url):
    params = {"type": "log", "key": api_key}
    url = f"{base_url}/api/"
    resp = requests.get(url, params=params, verify=False)
    if resp.status_code == 200:
        try:
            data = resp.json()
            logger.info("Successfully fetched Palo Alto logs via API.")
            return pd.DataFrame(data)
        except Exception as e:
            raise ValueError(f"Failed to parse Palo Alto logs from API: {e}")
    else:
        raise ValueError("Failed to fetch Palo Alto logs from API.")

def sonicwall_login(base_url, username, password):
    session = requests.Session()
    login_payload = {"username": username, "password": password}
    url = f"{base_url}/api/login"
    resp = session.post(url, json=login_payload, verify=False)
    if resp.status_code == 200:
        token = resp.json().get("token", "")
        logger.info("SonicWall login successful.")
        return session, token
    else:
        raise ValueError("SonicWall login failed.")

def fetch_sonicwall_logs(session, token, base_url):
    url = f"{base_url}/api/logs"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = session.get(url, headers=headers, verify=False)
    if resp.status_code == 200:
        try:
            data = resp.json()
            logger.info("Successfully fetched SonicWall logs via API.")
            return pd.DataFrame(data)
        except Exception as e:
            raise ValueError(f"Failed to parse SonicWall logs: {e}")
    else:
        raise ValueError("Failed to fetch SonicWall logs from API.")

def fetch_meraki_logs(api_key, base_url):
    headers = {"X-Cisco-Meraki-API-Key": api_key}
    url = f"{base_url}/api/v1/logs"
    resp = requests.get(url, headers=headers, verify=False)
    if resp.status_code == 200:
        try:
            data = resp.json()
            logger.info("Successfully fetched Meraki logs via API.")
            return pd.DataFrame(data)
        except Exception as e:
            raise ValueError(f"Failed to parse Meraki logs: {e}")
    else:
        raise ValueError("Failed to fetch Meraki logs from API.")

def determine_firewall_vendor(firewall_url):
    fw_lower = firewall_url.lower()
    if "forti" in fw_lower:
        return "FortiGate"
    elif "palo" in fw_lower or "pan" in fw_lower:
        return "PaloAlto"
    elif "sonic" in fw_lower:
        return "SonicWall"
    elif "meraki" in fw_lower:
        return "Meraki"
    elif "unifi" in fw_lower:
        return "Unifi"
    else:
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
    else:
        return ""

def load_firewall_logs(log_path):
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found at {log_path}")
    data = []
    with open(log_path, 'r') as f:
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
    logger.info("Loaded logs from local file.")
    return pd.DataFrame(data)

def preprocess_logs_for_model(df):
    df = df.copy()
    if "application" in df.columns:
        app_encoder = LabelEncoder()
        df["application_enc"] = app_encoder.fit_transform(df["application"].astype(str))
    else:
        df["application_enc"] = 0

    def ip_to_int(ip_str):
        try:
            parts = ip_str.split(".")
            return (int(parts[0]) << 24) + (int(parts[1]) << 16) + (int(parts[2]) << 8) + int(parts[3])
        except Exception:
            return 0
    if "src_ip" in df.columns:
        df["src_ip_int"] = df["src_ip"].apply(ip_to_int)
    elif "src" in df.columns:
        df["src_ip_int"] = df["src"].apply(ip_to_int)
    else:
        df["src_ip_int"] = 0
    if "dst_ip" in df.columns:
        df["dst_ip_int"] = df["dst_ip"].apply(ip_to_int)
    elif "dst" in df.columns:
        df["dst_ip_int"] = df["dst"].apply(ip_to_int)
    else:
        df["dst_ip_int"] = 0
    numeric_cols = ["src_ip_int", "dst_ip_int", "src_port", "dst_port", "application_enc"]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0
    data = df[numeric_cols].values.astype(np.float32)
    return data

def create_sequences(data, window_size=10):
    sequences = []
    for i in range(len(data) - window_size + 1):
        sequences.append(data[i:i+window_size])
    return np.array(sequences)

def filter_df_by_date_range(df, start_date, end_date, date_col="timestamp"):
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    return df[(df[date_col] >= start_date) & (df[date_col] <= end_date)].copy()

def build_enhanced_autoencoder(input_shape, latent_dim=32, dropout=0.3, learning_rate=1e-3):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = Bidirectional(LSTM(latent_dim, activation='relu', return_sequences=True))(x)
    x = Dropout(dropout)(x)
    x = Bidirectional(LSTM(latent_dim, activation='relu', return_sequences=True))(x)
    attention_output = Attention()([x, x])
    x = Concatenate()([x, attention_output])
    x = LSTM(latent_dim, activation='relu', return_sequences=False)(x)
    x = RepeatVector(input_shape[0])(x)
    x = LSTM(latent_dim, activation='relu', return_sequences=True)(x)
    x = Dropout(dropout)(x)
    x = LSTM(latent_dim, activation='relu', return_sequences=True)(x)
    outputs = TimeDistributed(Dense(input_shape[1]))(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate), loss='mse')
    logger.info("Enhanced BLSTM-Attention autoencoder built.")
    return model

def detect_anomalies_autoencoder(model, data, threshold):
    reconstructions = model.predict(data)
    mse = np.mean(np.square(data - reconstructions), axis=(1, 2))
    anomalies = mse > threshold
    return anomalies, mse

def get_anomaly_indices(anomalies_bool, window_size):
    anomaly_indices = set()
    for i, is_anom in enumerate(anomalies_bool):
        if is_anom:
            anomaly_indices.update(range(i, i + window_size))
    return sorted(anomaly_indices)

def get_anomaly_summary(df):
    src_ips = list(df["src_ip"].unique()) if "src_ip" in df.columns else []
    dst_ips = list(df["dst_ip"].unique()) if "dst_ip" in df.columns else []
    apps = list(df["application"].unique()) if "application" in df.columns else []
    return {"src_ips": src_ips, "dst_ips": dst_ips, "applications": apps}

def write_anomaly_report(df, output_filename="anomaly_report.json"):
    try:
        df.to_json(output_filename, orient="records", lines=True)
        logger.info(f"Anomaly report written to {output_filename}")
    except Exception as e:
        logger.error(f"Failed to write anomaly report: {e}")

def store_logs_in_redis(r, log_data, key_prefix="firewall_log"):
    try:
        with r.pipeline() as pipe:
            for log in log_data:
                log_id = uuid.uuid4().hex
                redis_key = f"{key_prefix}:{log_id}"
                pipe.hset(redis_key, mapping={k: str(v) for k, v in log.items()})
            pipe.execute()
        logger.info("Logs stored in Redis successfully.")
    except Exception as e:
        logger.error(f"Failed to store logs in Redis: {e}")

def retrieve_logs_from_redis(r, key_prefix="firewall_log"):
    try:
        keys = list(r.scan_iter(match=f"{key_prefix}:*"))
        log_entries = []
        for key in keys:
            entry = r.hgetall(key)
            decoded = {k.decode('utf-8'): v.decode('utf-8') for k, v in entry.items()}
            log_entries.append(decoded)
        logger.info("Logs retrieved from Redis successfully.")
        return pd.DataFrame(log_entries)
    except Exception as e:
        logger.error(f"Failed to retrieve logs from Redis: {e}")
        return pd.DataFrame()

def main_pipeline(
    firewall_url,
    baseline_start=None,
    baseline_end=None,
    forti_username="",
    forti_password="",
    palo_username="",
    palo_password="",
    sonic_username="",
    sonic_password="",
    meraki_api_key="",
    use_redis=False,
    redis_host="localhost",
    redis_port=6379,
    redis_db=0,
    window_size=10
):
    vendor = determine_firewall_vendor(firewall_url)
    df_logs = pd.DataFrame()
    try:
        if vendor == "FortiGate" and forti_username and forti_password:
            session, token = fortigate_login(firewall_url, forti_username, forti_password)
            df_logs = fetch_fortigate_logs(session, token, firewall_url)
        elif vendor == "PaloAlto" and palo_username and palo_password:
            api_key = paloalto_login(firewall_url, palo_username, palo_password)
            df_logs = fetch_paloalto_logs(api_key, firewall_url)
        elif vendor == "SonicWall" and sonic_username and sonic_password:
            session, token = sonicwall_login(firewall_url, sonic_username, sonic_password)
            df_logs = fetch_sonicwall_logs(session, token, firewall_url)
        elif vendor == "Meraki" and meraki_api_key:
            df_logs = fetch_meraki_logs(meraki_api_key, firewall_url)
    except Exception as e:
        logger.warning(f"API log retrieval failed: {e}. Falling back to local logs.")
    if df_logs.empty:
        log_path = get_log_location(vendor)
        if not log_path:
            logger.error("No valid log path found for vendor.")
            return
        df_logs = load_firewall_logs(log_path)
        if use_redis:
            r = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
            store_logs_in_redis(r, df_logs.to_dict(orient="records"))
    if df_logs.empty:
        logger.error("No logs to process.")
        return
    if "timestamp" in df_logs.columns:
        df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"], errors='coerce')
        df_logs.sort_values("timestamp", inplace=True)
    if baseline_start and baseline_end:
        try:
            s_date = pd.to_datetime(baseline_start)
            e_date = pd.to_datetime(baseline_end)
        except Exception as e:
            logger.error(f"Invalid baseline dates: {e}")
            return
        df_baseline = filter_df_by_date_range(df_logs, s_date, e_date)
        if len(df_baseline) < window_size:
            logger.error("Not enough baseline logs to create sequences.")
            return
        data_baseline = preprocess_logs_for_model(df_baseline)
    else:
        if len(df_logs) < window_size:
            logger.error("Not enough logs to create sequences.")
            return
        data_baseline = preprocess_logs_for_model(df_logs)
    scaler = MinMaxScaler()
    scaled_baseline = scaler.fit_transform(data_baseline)
    sequences_baseline = create_sequences(scaled_baseline, window_size=window_size)
    input_shape = (sequences_baseline.shape[1], sequences_baseline.shape[2])
    model = build_enhanced_autoencoder(input_shape=input_shape, latent_dim=64, dropout=0.3, learning_rate=1e-3)
    X_train, X_val = train_test_split(sequences_baseline, test_size=0.2, random_state=42)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=20,
        batch_size=64,
        callbacks=[es],
        verbose=1
    )
    recon_val = model.predict(X_val)
    mse_val = np.mean(np.square(X_val - recon_val), axis=(1,2))
    threshold = np.percentile(mse_val, 95)
    logger.info(f"Dynamic threshold: {threshold:.6f}")
    if baseline_start and baseline_end:
        df_outside = df_logs[~df_logs.index.isin(df_baseline.index)]
        if len(df_outside) >= window_size:
            data_outside = preprocess_logs_for_model(df_outside)
            scaled_outside = scaler.transform(data_outside)
            sequences_outside = create_sequences(scaled_outside, window_size)
            anom_bool, mse_vals = detect_anomalies_autoencoder(model, sequences_outside, threshold)
            idx_anom = get_anomaly_indices(anom_bool, window_size)
            df_anomalies = df_outside.iloc[idx_anom].drop_duplicates()
            logger.info(f"Found {len(df_anomalies)} anomalies outside baseline range.")
        else:
            logger.info("Insufficient logs outside baseline to detect anomalies.")
            df_anomalies = pd.DataFrame()
    else:
        anom_bool, mse_vals = detect_anomalies_autoencoder(model, sequences_baseline, threshold)
        idx_anom = get_anomaly_indices(anom_bool, window_size)
        df_anomalies = df_logs.iloc[idx_anom].drop_duplicates()
        logger.info(f"Found {len(df_anomalies)} anomalies in entire dataset.")
    if not df_anomalies.empty:
        anomaly_details = get_anomaly_summary(df_anomalies)
        logger.info(f"Anomaly details: {anomaly_details}")
        write_anomaly_report(df_anomalies, output_filename="anomaly_report.json")
    os.makedirs("model_data", exist_ok=True)
    model.save("model_data/enhanced_autoencoder.h5")
    with open("model_data/scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)
    with open("model_data/threshold.txt", "w") as fth:
        fth.write(str(threshold))
    logger.info("Saved model, scaler, and threshold in 'model_data/'.")

def incremental_train_on_new_logs():
    r = redis.Redis(
        host=os.environ.get("REDIS_HOST", "localhost"),
        port=int(os.environ.get("REDIS_PORT", 6379)),
        db=int(os.environ.get("REDIS_DB", 0))
    )
    model_path = "model_data/enhanced_autoencoder.h5"
    scaler_path = "model_data/scaler.pkl"
    threshold_path = "model_data/threshold.txt"
    if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(threshold_path):
        logger.error("Baseline model artifacts not found. Please run main_pipeline or baseline training first.")
        return
    logger.info("Loading model, scaler, and threshold for incremental training.")
    model = tf.keras.models.load_model(model_path, compile=True)
    with open(scaler_path, "rb") as fsc:
        scaler = pickle.load(fsc)
    with open(threshold_path, "r") as fth:
        threshold_str = fth.read().strip()
        threshold = float(threshold_str) if threshold_str else 0.0
    logger.info("Successfully loaded autoencoder, scaler, and threshold.")
    logger.info(f"Current threshold: {threshold:.6f}")
    channel_name = "new_log_batch"
    pubsub = r.pubsub()
    pubsub.subscribe(channel_name)
    logger.info(f"Subscribed to Redis channel: {channel_name}")
    buffer_size = 1000
    logs_buffer = []
    window_size = 10 
    incremental_epochs = 1 
    logger.info("Starting incremental training loop...")
    for message in pubsub.listen():
        if message["type"] != "message":
            continue
        try:
            new_logs = json.loads(message["data"])
            if isinstance(new_logs, list):
                logs_buffer.extend(new_logs)
            else:
                logs_buffer.append(new_logs)
        except Exception as e:
            logger.warning(f"Received invalid or non-JSON data: {message['data']}. Error: {e}")
            continue
        if len(logs_buffer) >= buffer_size:
            df_new = pd.DataFrame(logs_buffer)
            logs_buffer.clear()
            if df_new.empty:
                logger.info("Received an empty batch of logs; skipping training.")
                continue
            feats_new = preprocess_logs_for_model(df_new)
            scaled_new = scaler.transform(feats_new)
            if len(scaled_new) < window_size:
                logger.info("Not enough new data to form sequences for incremental training.")
                continue
            seq_new = create_sequences(scaled_new, window_size=window_size)
            logger.info(f"Performing incremental training on {len(seq_new)} sequences.")
            model.fit(seq_new, seq_new, epochs=incremental_epochs, batch_size=64, verbose=0)
            recon_new = model.predict(seq_new)
            mse_new = np.mean(np.square(seq_new - recon_new), axis=(1, 2))
            new_thresh = np.percentile(mse_new, 95)
            updated_thresh = (threshold + new_thresh) / 2.0
            model.save(model_path)
            with open(threshold_path, "w") as fth_upd:
                fth_upd.write(str(updated_thresh))
            threshold = updated_thresh
            logger.info(f"Incremental training done. Old threshold={threshold_str}, "
                        f"New threshold candidate={new_thresh:.6f}, "
                        f"Updated threshold ~ {threshold:.6f}")
            avg_mse = float(mse_new.mean())
            logger.info(f"New batch training stats: avg MSE={avg_mse:.6f}, 95th pct={new_thresh:.6f}")
