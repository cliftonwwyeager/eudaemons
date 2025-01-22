FROM python:3.9-slim
WORKDIR /eudaemons/eudaemons-docker
COPY eudaemons-flask.py .
RUN pip install --no-cache-dir tensorflow==2.9.1 numpy==1.21.6 protobuf==3.19.6 tensorflow-estimator==2.9.0
RUN pip install --no-cache-dir flask pandas scikit-learn scipy
ENV FIREWALL_URL=https://fortigate.example.com
EXPOSE 9500
CMD ["python", "eudaemons-flask.py", "--port", "9500"]
