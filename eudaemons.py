import os
import json
import argparse
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
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
from tensorflow.keras.layers import Dense, Dropout
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

def determine_firewall_vendor(firewall_url: str) -> str:
    fw_lower = firewall_url.lower()
    if "forti" in fw_lower:
        return "FortiGate"
    elif "palo" in fw_lower or "pan" in fw_lower:
        return "PaloAlto"
    else:
        return "Unknown"

def get_log_location(vendor: str) -> str:
    if vendor == "FortiGate":
        return "/var/log/fortigate/traffic.log"
    elif vendor == "PaloAlto":
        return "/var/log/paloalto/traffic.log"
    else:
        return ""

def load_firewall_logs(log_path: str) -> pd.DataFrame:
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found at {log_path}")
    with open(log_path, 'r') as f:
        data = []
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return pd.DataFrame(data)

def filter_df_by_date_range(df: pd.DataFrame, start_date, end_date, date_col="timestamp"):
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df_filtered = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)].copy()
    return df_filtered

def preprocess_logs_for_model(df: pd.DataFrame) -> np.ndarray:
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

def build_autoencoder(input_dim: int, latent_dim=16, dropout_rate=0.2) -> tf.keras.Model:
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(dropout_rate),
        Dense(latent_dim, activation='relu'),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(input_dim, activation='linear')
    ])
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

def train_autoencoder(model, X_train, X_val, epochs=10, batch_size=32):
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train, X_train, validation_data=(X_val, X_val),
              epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=1)
    return model

def calculate_dynamic_threshold(model: tf.keras.Model, X_val: np.ndarray, percentile=95) -> float:
    reconstructions_val = model.predict(X_val)
    mse_val = np.mean(np.square(X_val - reconstructions_val), axis=1)
    return np.percentile(mse_val, percentile)

def detect_anomalies_autoencoder(model, data, threshold):
    reconstructions = model.predict(data)
    mse = np.mean(np.square(data - reconstructions), axis=1)
    anomalies = mse > threshold
    return anomalies, mse

def get_anomaly_details(df: pd.DataFrame, anomalies: np.ndarray):
    anomaly_rows = df.iloc[anomalies.nonzero()[0]]
    src_ips = set(anomaly_rows["src_ip"].tolist()) if "src_ip" in df.columns else set()
    dst_ips = set(anomaly_rows["dst_ip"].tolist()) if "dst_ip" in df.columns else set()
    ports = set(anomaly_rows["dst_port"].tolist()) if "dst_port" in df.columns else set()
    apps = set(anomaly_rows["application"].tolist()) if "application" in df.columns else set()
    return {"src_ips": src_ips, "dst_ips": dst_ips, "ports": ports, "applications": apps}

def create_deny_rule_fortigate(firewall_url, session, token, anomalies, policy_id=9999, rule_name="Auto-Deny-Rule"):
    endpoint = f"{firewall_url}/api/v2/cmdb/firewall/policy/{policy_id}"
    headers = {"X-CSRFTOKEN": token, "Content-Type": "application/json"}
    payload = {
        "json": {
            "id": policy_id,
            "name": rule_name,
            "action": "deny",
            "srcaddr": [{"name": ip} for ip in anomalies["src_ips"]],
            "dstaddr": [{"name": ip} for ip in anomalies["dst_ips"]],
            "service": [{"name": f"TCP/{port}"} for port in anomalies["ports"]],
        }
    }
    try:
        resp = session.put(endpoint, headers=headers, json=payload, verify=False)
        if resp.status_code not in [200, 201]:
            print(f"[ERROR] FortiGate policy update failed: {resp.text}")
        else:
            print("[INFO] FortiGate deny rule updated/created successfully.")
    except Exception as e:
        print(f"[ERROR] FortiGate policy update exception: {e}")

def create_deny_rule_paloalto(firewall_url, api_key, anomalies, rule_name="Auto-Deny-Rule"):
    if panos is None:
        print("[ERROR] 'pan-os-python' library not installed.")
        return
    try:
        fw_host = firewall_url.replace("https://", "").replace("http://", "")
        fw = Firewall(hostname=fw_host, api_key=api_key)
        rule = SecurityRule(
            name=rule_name,
            fromzone=["any"],
            tozone=["any"],
            source=list(anomalies["src_ips"]),
            destination=list(anomalies["dst_ips"]),
            application=list(anomalies["applications"]),
            service=["any"],
            action="deny"
        )
        fw.add(rule)
        rule.create()
        print("[INFO] Palo Alto deny rule created/updated.")
    except Exception as e:
        print(f"[ERROR] Palo Alto rule push failed: {str(e)}")

def main_pipeline(firewall_url, baseline_start=None, baseline_end=None,
                  forti_username="", forti_password="",
                  palo_username="", palo_password=""):
    vendor = determine_firewall_vendor(firewall_url)
    print(f"[INFO] Detected vendor: {vendor}")
    log_path = get_log_location(vendor)
    if not log_path:
        print("[ERROR] Unknown vendor or no logs path available.")
        return

    df_logs = load_firewall_logs(log_path)
    print(f"[INFO] Loaded {len(df_logs)} log entries from {vendor} logs.")

    if baseline_start and baseline_end:
        try:
            s_date = pd.to_datetime(baseline_start)
            e_date = pd.to_datetime(baseline_end)
        except:
            print("[ERROR] Invalid date format for baseline.")
            return

        df_baseline = filter_df_by_date_range(df_logs, s_date, e_date)
        print(f"[INFO] Baseline date range: {baseline_start} to {baseline_end}, found {len(df_baseline)} entries.")
        if len(df_baseline) == 0:
            print("[ERROR] No baseline data in specified date range.")
            return

        X_baseline = preprocess_logs_for_model(df_baseline)
        X_train, X_val = train_test_split(X_baseline, test_size=0.2, random_state=42)

        model = build_autoencoder(input_dim=X_baseline.shape[1], latent_dim=8, dropout_rate=0.2)
        model = train_autoencoder(model, X_train, X_val, epochs=5, batch_size=64)

        # Compute threshold from validation set
        threshold = calculate_dynamic_threshold(model, X_val, percentile=95)

        df_outside_baseline = df_logs[~df_logs.index.isin(df_baseline.index)].copy()
        if len(df_outside_baseline) == 0:
            print("[INFO] No logs outside the baseline range.")
            return

        X_outside = preprocess_logs_for_model(df_outside_baseline)
        anomalies_bool, mse = detect_anomalies_autoencoder(model, X_outside, threshold)
        anomaly_details = get_anomaly_details(df_outside_baseline, anomalies_bool)
        print("[INFO] Anomaly details:", anomaly_details)

    else:
        X_all = preprocess_logs_for_model(df_logs)
        X_train, X_val = train_test_split(X_all, test_size=0.2, random_state=42)

        model = build_autoencoder(input_dim=X_all.shape[1], latent_dim=8, dropout_rate=0.2)
        model = train_autoencoder(model, X_train, X_val, epochs=5, batch_size=64)

        # Compute threshold from validation set
        threshold = calculate_dynamic_threshold(model, X_val, percentile=95)

        anomalies_bool, mse = detect_anomalies_autoencoder(model, X_all, threshold)
        anomaly_details = get_anomaly_details(df_logs, anomalies_bool)
        print("[INFO] Anomaly details:", anomaly_details)

    if vendor == "FortiGate":
        session, token = fortigate_login(firewall_url, forti_username, forti_password)
        create_deny_rule_fortigate(firewall_url, session, token, anomaly_details)
    elif vendor == "PaloAlto":
        api_key = paloalto_login(firewall_url, palo_username, palo_password)
        create_deny_rule_paloalto(firewall_url, api_key, anomaly_details)
    else:
        print("[ERROR] Firewall not recognized or not supported.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("-b", "--build-baseline", nargs=2, metavar=("START_DATE", "END_DATE"))
    parser.add_argument("--forti-username", default="")
    parser.add_argument("--forti-password", default="")
    parser.add_argument("--palo-username", default="")
    parser.add_argument("--palo-password", default="")
    args = parser.parse_args()

    if args.build_baseline:
        main_pipeline(
            firewall_url=args.url,
            baseline_start=args.build_baseline[0],
            baseline_end=args.build_baseline[1],
            forti_username=args.forti_username,
            forti_password=args.forti_password,
            palo_username=args.palo_username,
            palo_password=args.palo_password,
        )
    else:
        main_pipeline(
            firewall_url=args.url,
            forti_username=args.forti_username,
            forti_password=args.forti_password,
            palo_username=args.palo_username,
            palo_password=args.palo_password,
        )
