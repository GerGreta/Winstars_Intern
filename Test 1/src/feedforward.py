import torch
import torch.nn as nn
import torch.optim as optim
from interface import MnistClassifierInterface

# Реализация простой нейронной сети (Feed-Forward / MLP) для MNIST
class FeedForwardMnistClassifier(MnistClassifierInterface):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        # Определяем последовательную модель PyTorch
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # вход → скрытый слой
            nn.ReLU(),                            # активация ReLU
            nn.Linear(hidden_size, num_classes)   # скрытый слой → выход
        )
        # Функция потерь для многоклассовой классификации
        self.criterion = nn.CrossEntropyLoss()
        # Оптимизатор Adam
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    # Обучение сети
    def train(self, X_train, y_train, epochs=5):
        # Преобразуем данные в тензоры PyTorch
        X_train = torch.tensor(X_train.values if hasattr(X_train, "values") else X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train.values if hasattr(y_train, "values") else y_train, dtype=torch.long)

        # Проходим несколько эпох обучения
        for epoch in range(epochs):
            outputs = self.model(X_train)          # прямой проход
            loss = self.criterion(outputs, y_train) # вычисление потерь
            self.optimizer.zero_grad()             # обнуляем градиенты
            loss.backward()                        # обратное распространение
            self.optimizer.step()                  # шаг оптимизатора

    # Предсказание
    def predict(self, X_test):
        X_test = torch.tensor(X_test.values if hasattr(X_test, "values") else X_test, dtype=torch.float32)
        with torch.no_grad():                      # без вычисления градиентов
            outputs = self.model(X_test)
        return torch.argmax(outputs, dim=1).numpy() # выбираем класс с максимальным значением
