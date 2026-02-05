import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
import math

class SimpleModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out
    
    def string(self):
        return f'SimpleModel(fc1: {self.fc1}, fc2: {self.fc2})'
    
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
keptData = ['Pclass', 'Sex', 'Age', 'Parch', 'Fare']
xtrain_data = train_data[keptData]
ytrain_data = train_data['Survived']

xtrain_data = pd.get_dummies(xtrain_data).dropna()
xtest_data = pd.get_dummies(test_data[keptData]).dropna()
ytrain_data = ytrain_data.loc[xtrain_data.index]  # Ensure ytrain_data aligns with xtrain_data after dropping rows

xtrainnp = xtrain_data.to_numpy(dtype=np.float32)
ytrainnp = ytrain_data.to_numpy(dtype=np.float32)
xtestnp = xtest_data.to_numpy(dtype=np.float32)

xfinaltrain = torch.from_numpy(xtrainnp)
yfinaltrain = torch.from_numpy(ytrainnp)
xfinaltest = torch.from_numpy(xtestnp)

from sklearn.model_selection import train_test_split

xfinaltrain, xval, yfinaltrain, yval = train_test_split(xfinaltrain, yfinaltrain, test_size=0.15, random_state=42, shuffle=True)

# Determine input dimension from the processed training data
input_dimensions = xfinaltrain.shape[1]
output_dimensions = 1
model = SimpleModel(input_dimensions, 10, output_dimensions)

# Ensure target has shape (N, 1) for consistency with model outputs
if yfinaltrain.dim() == 1:
    yfinaltrain = yfinaltrain.unsqueeze(1)

# Standardize features (use training mean/std)
train_mean = xfinaltrain.mean(dim=0)
train_std = xfinaltrain.std(dim=0)
xfinaltrain = (xfinaltrain - train_mean) / (train_std + 1e-6)
xval = (xval - train_mean) / (train_std + 1e-6)
try:
    xfinaltest = (xfinaltest - train_mean) / (train_std + 1e-6)
except Exception:
    pass

train_losses = []

# Use a loss suited for binary classification and a modern optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

epochs = 200
for i in range(epochs):

    y_train_pred = model(xfinaltrain)

    train_loss = criterion(y_train_pred, yfinaltrain)

    if i % 20 == 0:
        print(f'Epoch {i}, Loss: {train_loss.item()}')
        train_losses.append(train_loss.item())

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    
print(f'Result: {model.string()}')

# Show predictions after training (probabilities and binary predictions)
model.eval()
with torch.no_grad():
    y_train_pred = model(xfinaltrain)
    probs = torch.sigmoid(y_train_pred)
    preds = (y_train_pred > 0.5).int()

    print('raw outputs (first 5):')
    print(y_train_pred[:5])
    print('probabilities (first 5):')
    print(probs[:5].squeeze())
    print('binary predictions (first 5):')
    print(preds[:5].squeeze())
