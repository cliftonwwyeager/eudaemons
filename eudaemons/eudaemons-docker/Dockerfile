FROM python:3.9-slim
WORKDIR /eudaemons/eudaemons-docker
COPY eudaemons-flask.py .
RUN pip install --no-cache-dir pandas flask requests scapy scikit-learn tensorflow==2.9.1 numpy==1.21.6
RUN pip install --no-cache-dir scipy
ENV FIREWALL_URL=https://fortigate.example.com
EXPOSE 5000
CMD ["python", "eudaemons-flask.py", "--port", "5000"]

