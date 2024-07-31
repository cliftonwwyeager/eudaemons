import subprocess
from elasticsearch import Elasticsearch
from fortiosapi import FortiOSAPI

ELASTICSEARCH_HOST = 'localhost'
ELASTICSEARCH_PORT = 9200
FORTIGATE_IP = '192.168.1.100'
FORTIGATE_USERNAME = 'admin'
FORTIGATE_PASSWORD = 'password'
FORTIGATE_VDOM = 'root'

es = Elasticsearch([{'host': ELASTICSEARCH_HOST, 'port': ELASTICSEARCH_PORT}])
fortigate = FortiOSAPI()

fortigate.login(FORTIGATE_IP, FORTIGATE_USERNAME, FORTIGATE_PASSWORD, vdom=FORTIGATE_VDOM)

def train_model():
    try:
        result = subprocess.check_output(['python', 'eudaemons-train.py'], universal_newlines=True)
        return result
    except subprocess.CalledProcessError as e:
        return str(e)

def export_to_elk(data):
    """
    Export data to an ELK stack.

    Args:
        data (dict): The data to export, typically a dictionary of anomaly details.

    Returns:
        str: A message indicating the result of the export operation.
    """
    try:
        response = es.index(index="anomalies", body=data)
        return f"Data exported to ELK: {response['_id']}"
    except Exception as e:
        return f"Failed to export data to ELK: {str(e)}"

def block_ip_fortigate(ip_address):
    """
    Block a given IP address on the Fortigate firewall.

    Args:
        ip_address (str): The IP address to block.

    Returns:
        str: A message indicating the result of the block operation.
    """
    try:
        rule = {
            "name": f"block_{ip_address}",
            "srcintf": [{"name": "any"}],
            "dstintf": [{"name": "any"}],
            "srcaddr": [{"name": ip_address}],
            "action": "deny",
            "schedule": "always",
            "service": [{"name": "ALL"}],
            "logtraffic": "all",
            "comments": "Blocked due to detected anomaly",
            "status": "enable",
            "policyid": None
        }

        response = fortigate.monitor('firewall.policy', 'set', vdom=FORTIGATE_VDOM, data=rule)
        return f"IP {ip_address} blocked on Fortigate: {response['status']}"
    except Exception as e:
        return f"Failed to block IP {ip_address} on Fortigate: {str(e)}"

def get_anomalies():
    """
    Fetch anomalies from ELK.

    Returns:
        list: A list of anomalies, each represented as a dictionary.
    """
    try:
        response = es.search(index="anomalies", body={"query": {"match_all": {}}})
        return response['hits']['hits']
    except Exception as e:
        return f"Failed to fetch anomalies from ELK: {str(e)}"

def get_blocked_ips():
    """
    Fetch blocked IPs from Fortigate.

    Returns:
        list: A list of blocked IP addresses, each represented as a dictionary.
    """
    try:
        response = fortigate.get('firewall', 'policy', vdom=FORTIGATE_VDOM)
        blocked_ips = [
            {
                "ip": policy['srcaddr'][0]['name'],
                "reason": policy['comments']
            }
            for policy in response['results']
            if policy['action'] == 'deny'
        ]
        return blocked_ips
    except Exception as e:
        return f"Failed to fetch blocked IPs from Fortigate: {str(e)}"