document.addEventListener('DOMContentLoaded', function() {
    const terminalOutput = document.getElementById('training-output');
    const trainButton = document.getElementById('train-btn');

    trainButton.addEventListener('click', function() {
        trainModel();
    });

    if (window.location.pathname === '/anomalies') {
        fetchAnomalies();
    }

    if (window.location.pathname === '/block_ips') {
        fetchBlockedIPs();
    }

    function trainModel() {
        terminalOutput.innerText = "Training model...";
        fetch('/train', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                terminalOutput.innerText = data.output;
            })
            .catch(error => {
                console.error('Error:', error);
                terminalOutput.innerText = 'An error occurred during training.';
            });
    }

    function fetchAnomalies() {
        fetch('/anomalies', { method: 'GET' })
            .then(response => response.json())
            .then(data => {
                displayAnomalies(data);
            })
            .catch(error => {
                console.error('Error:', error);
                displayError('Failed to fetch anomalies.');
            });
    }

    function fetchBlockedIPs() {
        fetch('/block_ips', { method: 'GET' })
            .then(response => response.json())
            .then(data => {
                displayBlockedIPs(data);
            })
            .catch(error => {
                console.error('Error:', error);
                displayError('Failed to fetch blocked IPs.');
            });
    }

    function displayAnomalies(anomalies) {
        const anomaliesList = document.getElementById('anomalies-list');
        anomaliesList.innerHTML = '';
        anomalies.forEach(anomaly => {
            const anomalyItem = document.createElement('div');
            anomalyItem.className = 'anomaly-item';
            anomalyItem.innerHTML = `
                <div class="anomaly-header">
                    <div><strong>ID:</strong> ${anomaly.id}</div>
                    <div><strong>Description:</strong> ${anomaly.description}</div>
                    <button class="expand-btn" data-id="${anomaly.id}">Expand</button>
                </div>
                <div class="anomaly-details" id="details-${anomaly.id}" style="display:none;">
                    <pre>${JSON.stringify(anomaly.details, null, 2)}</pre>
                    <div><strong>Terminal Output:</strong> ${anomaly.terminal_output}</div>
                    <div><strong>API Response Code:</strong> ${anomaly.api_response_code}</div>
                    <input type="checkbox" id="validate-${anomaly.id}" name="validate-${anomaly.id}">
                    <label for="validate-${anomaly.id}"> Validate</label>
                </div>
            `;
            anomaliesList.appendChild(anomalyItem);
        });

        document.querySelectorAll('.expand-btn').forEach(button => {
            button.addEventListener('click', function() {
                const id = this.dataset.id;
                const details = document.getElementById(`details-${id}`);
                details.style.display = details.style.display === 'none' ? 'block' : 'none';
            });
        });
    }

    function displayBlockedIPs(blockedIPs) {
        const blockedIPsList = document.getElementById('blocked-ips-list');
        blockedIPsList.innerHTML = '';
        blockedIPs.forEach(ip => {
            const ipItem = document.createElement('div');
            ipItem.className = 'ip-item';
            ipItem.innerHTML = `
                <div class="ip-header">
                    <div><strong>IP:</strong> ${ip.ip}</div>
                    <div><strong>Reason:</strong> ${ip.reason}</div>
                    <button class="expand-btn" data-ip="${ip.ip}">Expand</button>
                </div>
                <div class="ip-details" id="details-${ip.ip}" style="display:none;">
                    <pre>${JSON.stringify(ip, null, 2)}</pre>
                    <div><strong>Terminal Output:</strong> ${ip.terminal_output}</div>
                    <div><strong>API Response Code:</strong> ${ip.api_response_code}</div>
                    <input type="checkbox" id="validate-${ip.ip}" name="validate-${ip.ip}">
                    <label for="validate-${ip.ip}"> Validate</label>
                </div>
            `;
            blockedIPsList.appendChild(ipItem);
        });

        document.querySelectorAll('.expand-btn').forEach(button => {
            button.addEventListener('click', function() {
                const ip = this.dataset.ip;
                const details = document.getElementById(`details-${ip}`);
                details.style.display = details.style.display === 'none' ? 'block' : 'none';
            });
        });
    }

    function displayError(message) {
        const errorContainer = document.createElement('div');
        errorContainer.className = 'error';
        errorContainer.innerText = message;
        document.body.appendChild(errorContainer);
    }
});