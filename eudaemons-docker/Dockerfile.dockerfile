FROM python:3.9-slim
WORKDIR /eudaemons
COPY eudaemons.py /eudaemons-docker/
COPY trained_model.h5 /eudaemons-docker/
COPY scaler.pkl /eudaemons-docker/
RUN pip install --no-cache-dir flask requests scapy pandas numpy scikit-learn tensorflow==2.9.1
RUN pip install --no-cache-dir scipy
ENV FIREWALL_URL=https://fortigate.example.com
EXPOSE 9500
CMD ["python", "app.py", "--port", "9500"]
