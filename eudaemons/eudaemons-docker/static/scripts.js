function postAction(url) {
    fetch(url, {
        method: 'POST'
    }).then(response => {
        if(response.redirected) {
            window.location.href = response.url;
        } else {
            location.reload();
        }
    }).catch(error => console.error('Error:', error));
}

function fetchMetrics() {
    fetch('/metrics')
    .then(response => response.json())
    .then(data => {
        let anomaliesBox = document.getElementById('anomalies-box');
        anomaliesBox.innerHTML = '';
        data.anomalies.forEach(item => {
            let div = document.createElement('div');
            div.textContent = item;
            anomaliesBox.appendChild(div);
        });
        let performanceBox = document.getElementById('performance-box');
        performanceBox.innerHTML = '';
        data.performance.forEach(item => {
            let div = document.createElement('div');
            div.textContent = item;
            performanceBox.appendChild(div);
        });
        let exportedBox = document.getElementById('exported-box');
        exportedBox.innerHTML = '';
        data.exported.forEach(item => {
            let div = document.createElement('div');
            div.textContent = item;
            exportedBox.appendChild(div);
        });
    })
    .catch(error => console.error('Error fetching metrics:', error));
}

setInterval(fetchMetrics, 10000);

let ctx = document.getElementById('anomalyChart').getContext('2d');
let anomalyChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Anomaly Score',
            data: [], // Anomaly scores data
            borderColor: 'rgba(255, 99, 132, 1)',
            fill: false,
            tension: 0.1
        }]
    },
    options: {
        scales: {
            x: {
                type: 'time',
                time: {
                    unit: 'minute'
                }
            },
            y: {
                beginAtZero: true
            }
        }
    }
});

function updateChart() {
    fetch('/metrics')
        .then(response => response.json())
        .then(data => {
            let now = new Date();
            let anomalyCount = data.anomalies.length;
            anomalyChart.data.labels.push(now);
            anomalyChart.data.datasets[0].data.push(anomalyCount);
            if (anomalyChart.data.labels.length > 20) {
                anomalyChart.data.labels.shift();
                anomalyChart.data.datasets[0].data.shift();
            }
            anomalyChart.update();
        })
        .catch(error => console.error("Error updating chart:", error));
}

setInterval(updateChart, 10000);
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="{{ url_for('static', filename='scripts.js') }}"></script>
</body>
</html>
