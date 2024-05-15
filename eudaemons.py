import scapy.all as scapy
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Nadam
from deap import base, creator, tools, algorithms
import random
import numpy as np
import pickle

def ip_to_int(ip):
    return int.from_bytes(scapy.inet_aton(ip), 'big')

def capture_packets():
    packets = scapy.sniff(count=100000, filter="tcp")
    raw_data = []
    for packet in packets:
        if scapy.IP in packet and scapy.TCP in packet:
            payload = bytes(packet[scapy.TCP].payload) if packet[scapy.TCP].payload else b''
            application_protocol = packet[scapy.TCP].dport if packet[scapy.TCP].dport in [80, 443, 22, 3389, 5900, 21, 8080, 8443, 389, 139, 445, 2375, 1433] else 0  
            packet_dict = {
                "src": ip_to_int(packet[scapy.IP].src),
                "dst": ip_to_int(packet[scapy.IP].dst),
                "ttl": packet[scapy.IP].ttl,
                "sport": packet[scapy.TCP].sport,
                "dport": packet[scapy.TCP].dport,
                "payload": int.from_bytes(payload[:4], 'big') if payload else 0,
                "application_protocol": application_protocol
            }
            raw_data.append(packet_dict)
    return pd.DataFrame(raw_data)

def build_network(input_dim, params):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(params['first_layer'], activation='relu'),
        Dropout(params['dropout_rate']),
        Dense(params['second_layer'], activation='relu'),
        Dropout(params['dropout_rate']),
        Dense(1, activation='sigmoid')
    ])
    optimizer = Nadam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def eval_network(individual):
    params = dict(zip(['first_layer', 'second_layer', 'dropout_rate', 'learning_rate'], individual))
    model = build_network(7, params)
    packets = capture_packets()
    if not packets.empty:
        scaler = StandardScaler()
        data = scaler.fit_transform(packets[['src', 'dst', 'ttl', 'sport', 'dport', 'payload', 'application_protocol']])
        split_idx = int(len(data) * 0.8)
        train_data, train_labels = data[:split_idx], np.random.randint(2, size=split_idx)
        test_data, test_labels = data[split_idx:], np.random.randint(2, size=len(data) - split_idx)
        steps_per_epoch = len(train_data) // 10
        model.fit(train_data, train_labels, epochs=10, batch_size=10, steps_per_epoch=steps_per_epoch, verbose=0)
        _, accuracy = model.evaluate(test_data, test_labels, verbose=0)
        return (accuracy,)
    return (0,)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_first_layer", random.choice, [64, 128, 256])
toolbox.register("attr_second_layer", random.choice, [32, 64, 128])
toolbox.register("attr_dropout_rate", random.uniform, 0.1, 0.5)
toolbox.register("attr_learning_rate", random.uniform, 0.0001, 0.01)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_first_layer, toolbox.attr_second_layer,
                  toolbox.attr_dropout_rate, toolbox.attr_learning_rate), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", eval_network)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutPolynomialBounded, low=[64, 32, 0.1, 0.0001], up=[256, 128, 0.5, 0.01], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    population = toolbox.population(n=10)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=True)
    best_individual = tools.selBest(population, k=1)[0]
    best_params = dict(zip(['first_layer', 'second_layer', 'dropout_rate', 'learning_rate'], best_individual))
    final_model = build_network(7, best_params)
    final_scaler = StandardScaler()
    data = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0]] * 10, columns=['src', 'dst', 'ttl', 'sport', 'dport', 'payload', 'application_protocol'])
    final_scaler.fit(data)
    final_model.save('final_model.h5')
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(final_scaler, f)

if __name__ == "__main__":
    main()
