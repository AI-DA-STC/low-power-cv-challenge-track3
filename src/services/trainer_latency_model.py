import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from ann import LatencyANN

class LatencyTrainer:
    def __init__(self, X_train, Y_train, X_val, Y_val, epochs=50, batch_size=16, lr=0.001):
        self.device = "cuda" if torch.cuda.is_available() else "mps"
        self.model = LatencyANN(input_size=X_train.shape[1]).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Dataloaders
        self.train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)),
            batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                          torch.tensor(Y_val, dtype=torch.float32).view(-1, 1)),
            batch_size=batch_size, shuffle=False
        )
        self.epochs = epochs

    def train(self):
        """Train the ANN model."""
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for X_batch, Y_batch in self.train_loader:
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, Y_batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            val_loss = self.validate()
            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")

    def validate(self):
        """Validate the ANN model."""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in self.val_loader:
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, Y_batch)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

