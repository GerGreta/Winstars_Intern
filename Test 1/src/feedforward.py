from tensorflow import keras
from tensorflow.keras import layers
from interface import MnistClassifierInterface

class FeedForwardKerasClassifier(MnistClassifierInterface):
    def __init__(self, input_dim=784, num_classes=10):
        self.model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes, activation="softmax")
        ])
        self.model.compile(optimizer="adam",
                           loss="sparse_categorical_crossentropy",
                           metrics=["accuracy"])

    def train(self, X_train, y_train, epochs=5, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, X):
        preds = self.model.predict(X, verbose=0)
        return preds.argmax(axis=1)
