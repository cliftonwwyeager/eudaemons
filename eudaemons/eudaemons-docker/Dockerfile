FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY eudaemons.py /app/
COPY eudaemons-flask.py /app/
COPY static/ /app/static/

EXPOSE 9500

CMD ["python", "eudaemons-flask.py", "--port", "9500"]
