## eudaemons v1.10.4

This application provides a comprehensive pipeline for detecting anomalies in network firewall logs using an enhanced CNN-LSTM autoencoder model. It supports multiple firewall vendors including FortiGate, Palo Alto, SonicWall, Meraki, and Unifi. Can export detected anomalies to various SIEM platforms (Elastic, Cortex, Splunk, Sentinel).

## Features

- **CNN-LSTM Neural Network:**  
  Uses an enhanced autoencoder combining convolutional and stacked LSTM layers for improved anomaly detection.

- **Multi-Vendor Support:**  
  Native support for FortiGate, Palo Alto, SonicWall, Meraki and Unifi firewalls.

- **Redis Integration:**  
  Optionally cache logs using Redis.

- **SIEM Export:**  
  Automatically export detected anomalies to SIEM solutions like Elastic, Cortex, Splunk, and Sentinel.

- **Flask Web Interface:**  
  A simple web UI to trigger analysis and export of anomalies.

- **Docker Compose Ready:**  
  Easily build and deploy the entire application via Docker Compose.

## Requirements

All required Python dependencies are listed in the [`requirements.txt`](./requirements.txt) file. To install them, run:

`pip install -r requirements.txt`

## Docker Compose Setup

Build the Docker Image and Start the Services:
    `docker-compose up --build`
    
    Access the Web Interface:
    
    Open your browser and navigate to http://localhost:9500.

## Application Usage

-    **Analyze Firewall Logs:**
    Click the "Analyze Firewall Logs" button to run the anomaly detection pipeline on your firewall logs.

-    **Export Anomalies:**
    Click the "Export Anomalies to SIEM" button to send the detected anomalies to your configured SIEM platform.


## Notes

    Ensure the necessary log files and model files (enhanced_autoencoder.h5 and scaler.pkl) are present in the expected locations.
    Adjust environment variables in the docker-compose.yml file as needed to match your environment and SIEM configuration.
    For any issues or contributions, please open an issue or submit a pull request.
