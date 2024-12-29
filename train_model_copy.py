# train_model.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# Carregar dados
df = pd.read_csv('resultado_megasena.csv', usecols=lambda col: col.startswith('bola '))  # Ler apenas as bolas
scaler = MinMaxScaler()
data = scaler.fit_transform(df.values.astype(np.float32))

# Modelo Melhorado para Mega-Sena
class EnhancedNN(nn.Module):
    def __init__(self):
        super(EnhancedNN, self).__init__()
        self.fc1 = nn.Linear(6, 256)  # Ajustado para 6 bolas
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 6)  # Saída com 6 bolas
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Função de Treinamento
def train_model(model, data, targets, optimizer, criterion, epochs=10000):
    for epoch in range(epochs):
        inputs = torch.tensor(data, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
    return model

# Treinar Modelos
models = []
criterion = nn.MSELoss()

for i in range(6):
    model = EnhancedNN()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    trained_model = train_model(model, data, data[-1], optimizer, criterion)
    models.append(trained_model)
    torch.save(trained_model.state_dict(), f"megasena_model_{i+1}.pth")
    print(f"Modelo {i+1} treinado e salvo!")

# Função de Predição
def predict_numbers(model, data_tensor):
    inputs = torch.tensor(data_tensor, dtype=torch.float32)
    prediction = model(inputs).detach().numpy()[-1]
    return np.clip(np.round(prediction), 1, 60).astype(int)

# Garantir Diversidade
def is_diverse(pred1, pred2, threshold=4):
    return len(set(pred1) - set(pred2)) >= threshold

# Previsão Final
predictions = [predict_numbers(model, data) for model in models]

# Seleção de Conjuntos Diversos
final_predictions = [predictions[0]]

for pred in predictions[1:]:
    if all(is_diverse(pred, chosen) for chosen in final_predictions):
        final_predictions.append(pred)
        if len(final_predictions) == 6:
            break

# Exibir Resultados Finais
for i, pred in enumerate(final_predictions, 1):
    print(f"Previsão Final {i}: {sorted(pred)}")
