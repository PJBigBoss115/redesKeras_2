import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import Zeros, RandomNormal, GlorotUniform
import matplotlib.pyplot as plt

# Cargar datos y preprocesar
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

# Inicializadores
zero_initializer = Zeros()
normal_initializer = RandomNormal()
glorot_initializer = GlorotUniform()

# Modelos con diferentes inicializaciones
model_zeros = Sequential([
    Dense(128, activation='relu', input_shape=(784,), kernel_initializer=zero_initializer),
    Dense(128, activation='relu', kernel_initializer=zero_initializer),
    Dense(128, activation='relu', kernel_initializer=zero_initializer),
    Dense(10, activation='softmax', kernel_initializer=zero_initializer)
])

model_normal = Sequential([
    Dense(128, activation='relu', input_shape=(784,), kernel_initializer=normal_initializer),
    Dense(128, activation='relu', kernel_initializer=normal_initializer),
    Dense(128, activation='relu', kernel_initializer=normal_initializer),
    Dense(10, activation='softmax', kernel_initializer=normal_initializer)
])

model_glorot = Sequential([
    Dense(128, activation='relu', input_shape=(784,), kernel_initializer=glorot_initializer),
    Dense(128, activation='relu', kernel_initializer=glorot_initializer),
    Dense(128, activation='relu', kernel_initializer=glorot_initializer),
    Dense(10, activation='softmax', kernel_initializer=glorot_initializer)
])

# Compilación de modelos
model_zeros.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_normal.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_glorot.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento de modelos
history_zeros = model_zeros.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=0)
history_normal = model_normal.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=0)
history_glorot = model_glorot.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=0)

# Visualización de resultados
def plot_results(history, title):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(title + ' - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(title + ' - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

plot_results(history_zeros, "Zeros Initialization")
plot_results(history_normal, "Random Normal Initialization")
plot_results(history_glorot, "Glorot Uniform Initialization")