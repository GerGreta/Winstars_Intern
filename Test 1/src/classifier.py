from random_forest import RandomForestMnistClassifier
from feedforward import FeedForwardMnistClassifier
from cnn import CnnMnistClassifier

class MnistClassifier:
    def __init__(self, algorithm):
        if algorithm == "rf":
            self.classifier = RandomForestMnistClassifier()
        elif algorithm == "nn":
            self.classifier = FeedForwardMnistClassifier()
        elif algorithm == "cnn":
            self.classifier = CnnMnistClassifier()
        else:
            raise ValueError("Unknown algorithm: choose from ['rf', 'nn', 'cnn']")

    def train(self, X_train, y_train):
        self.classifier.train(X_train, y_train)

    def predict(self, X_test):
        return self.classifier.predict(X_test)
