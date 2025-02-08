#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt

# Load the Dataset
final_data = pd.read_csv("adjusted_final_data.csv")

# Ensure 'date' is in datetime format
final_data['date'] = pd.to_datetime(final_data['date'])

# Feature Engineering
final_data['month'] = final_data['date'].dt.month
final_data['day_of_year'] = final_data['date'].dt.dayofyear

# Select Features and Target
features = ['sla', 'month', 'day_of_year']  # Add more features if needed
target = 'anomaly'

X = final_data[features].fillna(0)  # Handle missing values
y = final_data[target]

# Normalize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create PyTorch Dataset Class
class AnomalyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = AnomalyDataset(X_train, pd.Series(y_train))
test_dataset = AnomalyDataset(X_test, pd.Series(y_test))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the Neural Network Model
class AnomalyModel(nn.Module):
    def __init__(self, input_size):
        super(AnomalyModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),        
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),          
            nn.ReLU(),
            nn.Linear(32, 1),         
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# Initialize the Model
input_size = X_train.shape[1]
model = AnomalyModel(input_size)

# Define Loss Function and Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move Model to Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training the Model
epochs = 250
losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward Pass
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)

        # Backward Pass and Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Plot Training Loss
plt.plot(range(1, epochs+1), losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.show()

# Evaluate the Model
model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs).squeeze()
        predictions.extend(outputs.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Convert Probabilities to Binary Predictions
binary_predictions = np.round(predictions)

torch.save(model.state_dict(), "anomaly_model.pth")
print("Model saved as anomaly_model.pth")

