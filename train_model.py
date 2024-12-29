# train_model.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Carregar dados
df = pd.read_csv('resultado_megasena.csv', usecols=lambda col: col.startswith('bola '))  # Ler apenas as bolas
scaler = MinMaxScaler()
data = scaler.fit_transform(df.values.astype(np.float32))

# Definir targets: Neste caso, usamos os dados com um deslocamento
targets = data[1:, :]  # Ajuste conforme necessário (por exemplo, data[1:] ou algo similar)

# Definir a classe do modelo
class EnhancedNN(nn.Module):
    def __init__(self):
        super(EnhancedNN, self).__init__()
        self.fc1 = nn.Linear(6, 512)   # Mais neurônios
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 6)    # Saída com 6 bolas
        self.dropout = nn.Dropout(0.4)  # Dropout aumentado

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Função de Treinamento com mini-batch
def train_model(model, data, targets, optimizer, criterion, epochs=10000, batch_size=64):
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Loss = {running_loss/len(dataloader):.6f}")
    return model

# Treinar o modelo
model = EnhancedNN()
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)  # Ajuste a taxa de aprendizado

trained_model = train_model(model, data[:-1], targets, optimizer, criterion, epochs=5000, batch_size=64)
torch.save(trained_model.state_dict(), "megasena_model.pth")
print("Modelo treinado e salvo!")
