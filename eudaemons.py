import dpkt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import SeparableConv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, GRU, TimeDistributed, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import tensorflow_model_optimization as tfmot
from scipy.stats import entropy, skew, kurtosis
from bayes_opt import BayesianOptimization

def extract_features(packet_list):
    if not packet_list:
        return np.zeros(12)
    packet_sizes = [len(packet[0]) for packet in packet_list]
    inter_arrival_times = np.diff([packet[1] for packet in packet_list])
    size_stats = [np.mean(packet_sizes), np.std(packet_sizes), np.min(packet_sizes), np.max(packet_sizes), skew(packet_sizes), kurtosis(packet_sizes)]
    inter_arrival_stats = [np.mean(inter_arrival_times), np.std(inter_arrival_times) if len(inter_arrival_times) > 0 else 0, skew(inter_arrival_times) if len(inter_arrival_times) > 0 else 0, kurtosis(inter_arrival_times) if len(inter_arrival_times) > 0 else 0]
    entropy_vals = [entropy(np.histogram(packet[0], bins=256)[0]) for packet in packet_list]
    feature_vector = size_stats + inter_arrival_stats + [np.mean(entropy_vals)]
    return np.array(feature_vector)

def process_pcap_to_bytes(file_path, max_packet_length=1500):
    packets = []
    labels = []
    packet_list = []
    with open(file_path, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        for timestamp, buf in pcap:
            eth = dpkt.ethernet.Ethernet(buf)
            if not isinstance(eth.data, dpkt.ip.IP):
                continue
            ip = eth.data
            if not isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                continue
            transport = ip.data
            packet_bytes = np.frombuffer(buf, dtype=np.uint8)
            if len(packet_bytes) > max_packet_length:
                packet_bytes = packet_bytes[:max_packet_length]
            elif len(packet_bytes) < max_packet_length:
                packet_bytes = np.pad(packet_bytes, (0, max_packet_length - len(packet_bytes)), 'constant')
            packet_list.append((packet_bytes, timestamp))
            if transport.sport == 21 or transport.dport == 22:
                labels.append(1)
            else:
                labels.append(0)
    packets = [extract_features(packet_list)]
    return np.array(packets), np.array(labels)

def build_cnn_gru_model(input_shape, learning_rate, dropout_rate, gru_units):
    input_layer = Input(shape=input_shape)
    x = TimeDistributed(SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu'))(input_layer)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
    x = TimeDistributed(Dropout(dropout_rate))(x)
    x = TimeDistributed(SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
    x = TimeDistributed(Dropout(dropout_rate))(x)
    x = TimeDistributed(SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
    x = TimeDistributed(Dropout(dropout_rate))(x)
    x = TimeDistributed(Flatten())(x)
    x = GRU(gru_units, activation='relu', return_sequences=True)(x)
    x = Dropout(dropout_rate)(x)
    x = GRU(gru_units // 2, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    output_layer = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def quantize_model(model):
    return tfmot.quantization.keras.quantize_model(model)

def train_on_gpu(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    with tf.device('/GPU:0'):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                  callbacks=[early_stopping, reduce_lr])
    return model

def process_and_train(file_path, learning_rate, dropout_rate, gru_units):
    packets, labels = process_pcap_to_bytes(file_path)
    packets = packets.reshape(-1, packets.shape[1], 1)
    X_train, X_test, y_train, y_test = train_test_split(packets, labels, test_size=0.2, random_state=42)
    cnn_gru_model = build_cnn_gru_model((packets.shape[1], packets.shape[2], 1), learning_rate, dropout_rate, gru_units)
    trained_cnn_gru_model = train_on_gpu(cnn_gru_model, X_train, y_train, X_test, y_test)
    loss, accuracy = trained_cnn_gru_model.evaluate(X_test, y_test)
    return loss, accuracy

def objective(learning_rate, dropout_rate, gru_units):
    file_path = 'path_to_pcap_file.pcap'
    loss, accuracy = process_and_train(file_path, learning_rate, dropout_rate, gru_units)
    return -accuracy

pbounds = {
    'learning_rate': (1e-5, 1e-2),
    'dropout_rate': (0.1, 0.5),
    'gru_units': (32, 256)
}

optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=42,
)

optimizer.maximize(
    init_points=2,
    n_iter=10,
)

print(optimizer.max)
