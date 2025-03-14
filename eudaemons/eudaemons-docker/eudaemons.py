import os
import json
import uuid
import argparse
import logging
from datetime import datetime
import requests
import numpy as np
import pandas as pd
import redis
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization, LSTM, RepeatVector, TimeDistributed, Conv1D)
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

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
    resp = session.get(url, headers=headers, verify=False)
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
        try:
            from xml.etree import ElementTree
            root = ElementTree.fromstring(resp.text)
            key = root.find("./result/key").text
            logger.info("Palo Alto login successful.")
            return key
        except Exception as e:
            raise ValueError(f"Failed to parse Palo Alto API key: {e}")
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
    else:
        return "Unknown"

def store_logs_in_redis(redis_host, redis_port, redis_db, log_data, key_prefix="firewall_log"):
    try:
        r = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        with r.pipeline() as pipe:
            for log in log_data:
                log_id = uuid.uuid4().hex
                redis_key = f"{key_prefix}:{log_id}"
                pipe.hset(redis_key, mapping={k: str(v) for k, v in log.items()})
            pipe.execute()
        logger.info("Logs stored in Redis successfully.")
    except Exception as e:
        logger.error(f"Failed to store logs in Redis: {e}")

def retrieve_logs_from_redis(redis_host, redis_port, redis_db, key_prefix="firewall_log"):
    try:
        r = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
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

def get_log_location(vendor):
    if vendor == "FortiGate":
        return "/var/log/fortigate/traffic.log"
    elif vendor == "PaloAlto":
        return "/var/log/paloalto/traffic.log"
    elif vendor == "SonicWall":
        return "/var/log/sonicwall/traffic.log"
    elif vendor == "Meraki":
        return "/var/log/meraki/traffic.log"
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
                data.append(json.loads(line))
    logger.info("Loaded logs from local file.")
    return pd.DataFrame(data)

def filter_df_by_date_range(df, start_date, end_date, date_col="timestamp"):
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    return df[(df[date_col] >= start_date) & (df[date_col] <= end_date)].copy()

def preprocess_logs_for_model(df):
    df = df.copy()
    # Encode application if available
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
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def create_sequences(data, window_size=10):
    sequences = [data[i:i + window_size] for i in range(len(data) - window_size + 1)]
    return np.array(sequences)

def build_enhanced_autoencoder(input_shape, latent_dim=32, dropout_rate=0.3, learning_rate=1e-3):
    """
    Builds an enhanced autoencoder model using a combination of Conv1D and stacked LSTM layers.
    """
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(LSTM(latent_dim, activation='relu', return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(latent_dim, activation='relu', return_sequences=False))
    model.add(RepeatVector(input_shape[0]))
    model.add(LSTM(latent_dim, activation='relu', return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(latent_dim, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(input_shape[1])))
    model.compile(optimizer=Adam(learning_rate), loss='mse')
    logger.info("Enhanced CNN–LSTM autoencoder built.")
    return model

def train_autoencoder(model, X_train, X_val, epochs=10, batch_size=32):
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train, X_train, validation_data=(X_val, X_val),
              epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=1)
    return model

def calculate_dynamic_threshold(model, X_val, percentile=95):
    reconstructions_val = model.predict(X_val)
    mse_val = np.mean(np.square(X_val - reconstructions_val), axis=(1, 2))
    threshold = np.percentile(mse_val, percentile)
    logger.info(f"Dynamic threshold calculated: {threshold:.4f}")
    return threshold

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
    src_ips = set(df["src_ip"].tolist()) if "src_ip" in df.columns else set()
    dst_ips = set(df["dst_ip"].tolist()) if "dst_ip" in df.columns else set()
    ports = set(df["dst_port"].tolist()) if "dst_port" in df.columns else set()
    apps = set(df["application"].tolist()) if "application" in df.columns else set()
    return {"src_ips": list(src_ips), "dst_ips": list(dst_ips), "ports": list(ports), "applications": list(apps)}

def write_anomaly_report(df, output_filename="anomaly_report.json"):
    try:
        df.to_json(output_filename, orient="records", lines=True)
        logger.info(f"Anomaly report written to {output_filename}")
    except Exception as e:
        logger.error(f"Failed to write anomaly report: {e}")

def main_pipeline(firewall_url,
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
                  window_size=10):
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
            store_logs_in_redis(redis_host, redis_port, redis_db, df_logs.to_dict(orient="records"))
    
    if df_logs.empty:
        logger.error("No logs to process.")
        return

    # Sort logs by timestamp if available
    if "timestamp" in df_logs.columns:
        df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"], errors='coerce')
        df_logs.sort_values("timestamp", inplace=True)

    # Determine training mode: baseline vs. all logs
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
        baseline_scaled = preprocess_logs_for_model(df_baseline)
        baseline_sequences = create_sequences(baseline_scaled, window_size=window_size)
        X_train, X_val = train_test_split(baseline_sequences, test_size=0.2, random_state=42)
        model = build_enhanced_autoencoder(input_shape=baseline_sequences.shape[1:])
        model = train_autoencoder(model, X_train, X_val, epochs=5, batch_size=64)
        threshold = calculate_dynamic_threshold(model, X_val, percentile=95)
        df_outside = df_logs[~df_logs.index.isin(df_baseline.index)]
        if len(df_outside) < window_size:
            logger.error("Not enough non-baseline logs to create sequences.")
            return
        outside_scaled = preprocess_logs_for_model(df_outside)
        outside_sequences = create_sequences(outside_scaled, window_size=window_size)
        anomalies_bool, mse = detect_anomalies_autoencoder(model, outside_sequences, threshold)
        anomaly_indices = get_anomaly_indices(anomalies_bool, window_size)
        df_anomalies = df_outside.iloc[anomaly_indices].drop_duplicates()
        anomaly_details = get_anomaly_summary(df_anomalies)
        logger.info(f"Anomaly details (non-baseline logs): {anomaly_details}")
    else:
        if len(df_logs) < window_size:
            logger.error("Not enough logs to create sequences.")
            return
        all_scaled = preprocess_logs_for_model(df_logs)
        all_sequences = create_sequences(all_scaled, window_size=window_size)
        X_train, X_val = train_test_split(all_sequences, test_size=0.2, random_state=42)
        model = build_enhanced_autoencoder(input_shape=all_sequences.shape[1:])
        model = train_autoencoder(model, X_train, X_val, epochs=5, batch_size=64)
        threshold = calculate_dynamic_threshold(model, X_val, percentile=95)
        anomalies_bool, mse = detect_anomalies_autoencoder(model, all_sequences, threshold)
        anomaly_indices = get_anomaly_indices(anomalies_bool, window_size)
        df_anomalies = df_logs.iloc[anomaly_indices].drop_duplicates()
        anomaly_details = get_anomaly_summary(df_anomalies)
        logger.info(f"Anomaly details (all logs): {anomaly_details}")
    write_anomaly_report(df_anomalies)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Network Traffic Anomaly Detection Pipeline")
    parser.add_argument("--url", required=True, help="Base URL of the firewall API or host")
    parser.add_argument("-b", "--build-baseline", nargs=2, metavar=("START_DATE", "END_DATE"),
                        help="Build baseline using logs between START_DATE and END_DATE")
    parser.add_argument("--forti-username", default="", help="FortiGate username")
    parser.add_argument("--forti-password", default="", help="FortiGate password")
    parser.add_argument("--palo-username", default="", help="Palo Alto username")
    parser.add_argument("--palo-password", default="", help="Palo Alto password")
    parser.add_argument("--sonic-username", default="", help="SonicWall username")
    parser.add_argument("--sonic-password", default="", help="SonicWall password")
    parser.add_argument("--meraki-api-key", default="", help="Meraki API key")
    parser.add_argument("--use-redis", action="store_true", help="Use Redis for log caching")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--redis-db", type=int, default=0, help="Redis database number")
    parser.add_argument("--window-size", type=int, default=10, help="Sliding window size for sequences")
    args = parser.parse_args()

    main_pipeline(
        firewall_url=args.url,
        baseline_start=args.build_baseline[0] if args.build_baseline else None,
        baseline_end=args.build_baseline[1] if args.build_baseline else None,
        forti_username=args.forti_username,
        forti_password=args.forti_password,
        palo_username=args.palo_username,
        palo_password=args.palo_password,
        sonic_username=args.sonic_username,
        sonic_password=args.sonic_password,
        meraki_api_key=args.meraki_api_key,
        use_redis=args.use_redis,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
        window_size=args.window_size
    )
