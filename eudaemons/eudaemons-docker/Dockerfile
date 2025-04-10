FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY eudaemons.py .
COPY eudaemons-flask.py .
COPY static/ ./static/
COPY templates/ ./templates/

EXPOSE 9500

CMD ["python", "eudaemons-flask.py", "--port", "9500"]
