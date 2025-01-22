async function analyzeLogs() {
  try {
    const response = await fetch("/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (response.ok) {
      const html = await response.text();
      document.body.innerHTML = html;
    } else {
      console.error("Error analyzing logs:", response.status, response.statusText);
      alert("Failed to analyze logs.");
    }
  } catch (error) {
    console.error("Error analyzing logs:", error);
    alert("An error occurred while analyzing logs.");
  }
}

async function exportAnomalies() {
  try {
    const response = await fetch("/export", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (response.ok) {
      const html = await response.text();
      document.body.innerHTML = html;
    } else {
      console.error("Error exporting anomalies:", response.status, response.statusText);
      alert("Failed to export anomalies.");
    }
  } catch (error) {
    console.error("Error exporting anomalies:", error);
    alert("An error occurred while exporting anomalies.");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const analyzeButton = document.querySelector("form[action='/analyze'] button");
  const exportButton = document.querySelector("form[action='/export'] button");

  if (analyzeButton) {
    analyzeButton.addEventListener("click", (event) => {
      event.preventDefault();
      analyzeLogs();
    });
  }

  if (exportButton) {
    exportButton.addEventListener("click", (event) => {
      event.preventDefault();
      exportAnomalies();
    });
  }
});
