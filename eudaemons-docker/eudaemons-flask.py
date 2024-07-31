from flask import Flask, request, jsonify
from threading import Thread
import pyshark
from utils import extract_features, setup_redis, stop_training_flag, set_stop_training_flag
from model import train_model

app = Flask(__name__)

# Setup Redis client
setup_redis()

@app.route('/train', methods=['POST'])
def train():
    interface = request.json.get('interface', 'eth0')
    # Reset the stop training flag
    set_stop_training_flag(False)
    # Start packet capture in a new thread
    capture_thread = Thread(target=capture_packets, args=(interface,))
    capture_thread.start()
    return jsonify({'status': 'training started and capturing packets'})

@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    # Set the stop training flag
    set_stop_training_flag(True)
    return jsonify({'status': 'capture stopped'})

def capture_packets(interface='eth0'):
    capture = pyshark.LiveCapture(interface=interface)
    for packet in capture.sniff_continuously():
        if stop_training_flag():
            break
        features = extract_features(packet)
        train_model(features)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)