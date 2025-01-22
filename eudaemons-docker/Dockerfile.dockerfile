FROM python:3.9-slim
WORKDIR /eudaemons/eudaemons-docker
RUN pip install --no-cache-dir pandas flask requests scapy scikit-learn tensorflow==2.9.1 numpy==1.21.6
RUN pip install --no-cache-dir scipy
COPY eudaemons-flask.py .
EXPOSE 9500
CMD ["python", "eudaemons-flask.py", "--port", "9500"]
