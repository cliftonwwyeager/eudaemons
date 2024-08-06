import os
import dpkt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer

class SyntheticDataGenerator:
    def __init__(self, max_packet_length=1500):
        self.max_packet_length = max_packet_length

    def generate_normal_traffic(self, num_packets):
        packets = []
        for _ in range(num_packets):
            packet_length = np.random.randint(40, self.max_packet_length)
            packet = np.random.randint(0, 256, packet_length, dtype=np.uint8)
            packet = np.pad(packet, (0, self.max_packet_length - len(packet)), 'constant')
            packets.append(packet)
        return np.array(packets), np.zeros(num_packets)

    def generate_anomalous_traffic(self, num_packets):
        packets = []
        for _ in range(num_packets):
            packet_length = np.random.randint(40, self.max_packet_length)
            packet = np.random.randint(0, 256, packet_length, dtype=np.uint8)
            packet = np.pad(packet, (0, self.max_packet_length - len(packet)), 'constant')
            packets.append(packet)
        return np.array(packets), np.ones(num_packets)

class CNN_GRU(nn.Module):
    def __init__(self, input_channels, dropout_rate, gru_units):
        super(CNN_GRU, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.gru = nn.GRU(128 * 16 * 16, gru_units, batch_first=True)
        self.fc1 = nn.Linear(gru_units, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        c_in = x.view(batch_size * time_steps, C, H, W)
        c_out = self.pool1(self.bn1(torch.relu(self.conv1(c_in))))
        c_out = self.pool2(self.bn2(torch.relu(self.conv2(c_out))))
        c_out = self.pool3(self.bn3(torch.relu(self.conv3(c_out))))
        r_in = c_out.view(batch_size, time_steps, -1)
        r_out, _ = self.gru(r_in)
        r_out = self.dropout(torch.relu(self.fc1(r_out[:, -1, :])))
        out = torch.sigmoid(self.fc2(r_out))
        return out

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def data_augmentation(data):
    augmented_data = []
    for packet in data:
        noise = np.random.normal(0, 0.01, packet.shape)
        augmented_data.append(packet + noise)
    return np.array(augmented_data)

def train(rank, world_size, data, labels, epochs, batch_size, learning_rate, dropout_rate, gru_units):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
    pca = PCA(n_components=100)
    X_train = pca.fit_transform(X_train.reshape(len(X_train), -1))
    X_val = pca.transform(X_val.reshape(len(X_val), -1))
    tfidf = TfidfTransformer()
    X_train = tfidf.fit_transform(X_train).toarray()
    X_val = tfidf.transform(X_val).toarray()
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = CNN_GRU(input_channels=1, dropout_rate=dropout_rate, gru_units=gru_units).to(device)
    model = DDP(model, device_ids=[rank])
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-6)
    best_accuracy = 0.0
    best_model_path = f"best_model_{rank}.pth"

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                running_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted.squeeze() == labels).sum().item()

        val_loss = running_loss / len(val_loader.dataset)
        accuracy = correct / total
        scheduler.step(val_loss)

        if rank == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}')
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), best_model_path)
                print(f'Saved Best Model with Accuracy: {best_accuracy:.4f}')

    dist.destroy_process_group()

def setup(rank, world_size, data, labels, epochs, batch_size, learning_rate, dropout_rate, gru_units):
    mp.spawn(train, args=(world_size, data, labels, epochs, batch_size, learning_rate, dropout_rate, gru_units), nprocs=world_size, join=True)

def main():
    input_channels = 1
    dropout_rate = 0.5
    gru_units = 128
    learning_rate = 0.001
    batch_size = 32
    epochs = 10
    world_size = 2
    generator = SyntheticDataGenerator()
    normal_packets, normal_labels = generator.generate_normal_traffic(10000)
    anomalous_packets, anomalous_labels = generator.generate_anomalous_traffic(2000)
    packets = np.concatenate((normal_packets, anomalous_packets), axis=0)
    labels = np.concatenate((normal_labels, anomalous_labels), axis=0)
    packets = packets.reshape(-1, 1, 1, 128, 128)
    labels = labels.astype(np.float32)
    packets = data_augmentation(packets)
    setup(0, world_size, packets, labels, epochs, batch_size, learning_rate, dropout_rate, gru_units)

if __name__ == "__main__":
    main()
