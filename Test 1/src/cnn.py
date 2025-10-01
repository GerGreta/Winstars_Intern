import torch
import torch.nn as nn
import torch.optim as optim
from interface import MnistClassifierInterface

class CnnMnistClassifier(MnistClassifierInterface):
    def __init__(self):
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, X_train, y_train, epochs=5):
        X_train = torch.tensor(X_train.values if hasattr(X_train, "values") else X_train, dtype=torch.float32).view(-1,1,28,28)
        y_train = torch.tensor(y_train.values if hasattr(y_train, "values") else y_train, dtype=torch.long)

        for epoch in range(epochs):
            outputs = self.model(X_train)
            loss = self.criterion(outputs, y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, X_test):
        X_test = torch.tensor(X_test.values if hasattr(X_test, "values") else X_test, dtype=torch.float32).view(-1,1,28,28)
        with torch.no_grad():
            outputs = self.model(X_test)
        return torch.argmax(outputs, dim=1).numpy()
