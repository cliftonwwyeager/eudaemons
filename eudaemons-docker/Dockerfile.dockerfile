FROM python:3.9-slim
WORKDIR /app
COPY app.py /app/
# Copy your model/scaler if present
# COPY trained_model.h5 /app/
# COPY scaler.pkl /app/

RUN pip install --no-cache-dir flask requests scapy pandas numpy scikit-learn tensorflow==2.9.1
RUN pip install --no-cache-dir scipy

ENV FIREWALL_URL=https://fortigate.example.com
EXPOSE 9500
CMD ["python", "app.py", "--port", "9500"]
