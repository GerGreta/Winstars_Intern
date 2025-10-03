from tensorflow import keras
from tensorflow.keras import layers
from interface import MnistClassifierInterface


class CnnMnistClassifier(MnistClassifierInterface):
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        """
        input_shape: форма одного изображения (28,28,1 для MNIST)
        num_classes: количество классов (10 для MNIST)
        """
        self.model = keras.Sequential([
            layers.Input(shape=input_shape),  # ✅ первый слой Input
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])

        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X_train, y_train, epochs=5, batch_size=32):
        """
        X_train: массив изображений (num_samples, 28, 28, 1)
        y_train: массив меток (num_samples,)
        """
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, X_test):
        """
        X_test: массив изображений (num_samples, 28, 28, 1)
        Возвращает массив предсказанных классов
        """
        preds = self.model.predict(X_test, verbose=0)
        return preds.argmax(axis=1)
