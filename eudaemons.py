import dpkt
import socket
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow_model_optimization as tfmot
import pickle

def process_pcap_to_bytes(file_path, max_packet_length=1500):
    packets = []
    with open(file_path, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        for timestamp, buf in pcap:
            packet_bytes = np.frombuffer(buf, dtype=np.uint8)
            if len(packet_bytes) > max_packet_length:
                packet_bytes = packet_bytes[:max_packet_length]
            elif len(packet_bytes) < max_packet_length:
                packet_bytes = np.pad(packet_bytes, (0, max_packet_length - len(packet_bytes)), 'constant')
            packets.append(packet_bytes)
    return np.array(packets)

def build_cnn_model(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv1D(filters=32, kernel_size=3, activation='relu')(input_layer)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

class ShiftBatchNormalization(Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(ShiftBatchNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=input_shape[-1:], initializer="ones", trainable=True)
        self.beta = self.add_weight(shape=input_shape[-1:], initializer="zeros", trainable=True)
        self.moving_mean = self.add_weight(shape=input_shape[-1:], initializer="zeros", trainable=False)
        self.moving_variance = self.add_weight(shape=input_shape[-1:], initializer="ones", trainable=False)

    def call(self, inputs, training=None):
        if training:
            mean = tf.reduce_mean(inputs, axis=0)
            variance = tf.reduce_mean(tf.square(inputs - mean), axis=0)
            self.moving_mean.assign(mean)
            self.moving_variance.assign(variance)
        else:
            mean = self.moving_mean
            variance = self.moving_variance

        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * normalized + self.beta

class ShiftAdamax(Optimizer):
    def __init__(self, learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-7, **kwargs):
        super(ShiftAdamax, self).__init__(name="ShiftAdamax", **kwargs)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "u")

    def _resource_apply_dense(self, grad, var, apply_state=None):
        lr_t = self.learning_rate
        beta_1_t = self.beta_1
        beta_2_t = self.beta_2
        epsilon_t = self.epsilon
        m = self.get_slot(var, "m")
        u = self.get_slot(var, "u")
        m_t = beta_1_t * m + (1 - beta_1_t) * grad
        u_t = tf.maximum(beta_2_t * u, tf.abs(grad))
        var_t = var - lr_t / (u_t + epsilon_t) * m_t      self._updates.append(tf.compat.v1.assign(var, var_t))       self._updates.append(tf.compat.v1.assign(m, m_t))  self._updates.append(tf.compat.v1.assign(u, u_t))

def quantize_model(model):
    return tfmot.quantization.keras.quantize_model(model)

def train_on_gpu(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    with tf.device('/GPU:0'):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),                 callbacks=[early_stopping, reduce_lr])
    return model

def process_and_train(file_path):
    packets = process_pcap_to_bytes(file_path)
    packets = packets.reshape(-1, packets.shape[1], 1)
    labels = np.random.randint(2, size=(packets.shape[0], 1))
    X_train, X_test, y_train, y_test = train_test_split(packets, labels, test_size=0.2, random_state=42)
    cnn_model = build_cnn_model((X_train.shape[1], 1))
    trained_cnn_model = train_on_gpu(cnn_model, X_train, y_train, X_test, y_test)  trained_cnn_model.save('optimized_cnn_model.h5')
    df = process_pcap(file_path)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[['timestamp', 'proto', 'length', 'flags', 'window_size']])
    labels = df['label'].values
    X = tf.constant(scaled_features, dtype=tf.float32)
    y = tf.constant(labels, dtype=tf.float32).unsqueeze(1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    q_model = build_and_train_model(X_train, y_train)
    q_model.save('optimized_bnn_model.h5')

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return trained_cnn_model, q_model

def infer_new_data(cnn_model, bnn_model, file_path):
    new_packets = process_pcap_to_bytes(file_path)
    new_packets = new_packets.reshape(-1, new_packets.shape[1], 1)
    cnn_predictions = cnn_model.predict(new_packets)
    new_data = process_pcap(file_path)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    scaled_new_data = scaler.transform(new_data[['timestamp', 'proto', 'length', 'flags', 'window_size']])
    X_new = tf.constant(scaled_new_data, dtype=tf.float32)
    bnn_predictions = bnn_model(X_new)
    anomalies = ((cnn_predictions > 0.5) | (bnn_predictions > 0.5)).astype(int)
    new_data['anomaly'] = anomalies
    print("Anomalies detected:", new_data[new_data['anomaly'] == 1])
cnn_model, bnn_model = process_and_train('path_to_pcap_file.pcap')
infer_new_data(cnn_model, bnn_model, 'new_pcap_file.pcap')