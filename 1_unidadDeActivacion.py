import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# Cargar datos y preprocesar
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0  # Asegurarse de tener la forma correcta (784 características)
x_test = x_test.reshape(-1, 784) / 255.0

# Función para comparar la precisión de dos entrenamientos
def plot_compare_accs(history1, history2, name1="Model 1", name2="Model 2", title="Comparison of Accuracy"):
    plt.plot(history1.history['accuracy'], color="green")
    plt.plot(history1.history['val_accuracy'], 'r--', color="green")
    plt.plot(history2.history['accuracy'], color="blue")
    plt.plot(history2.history['val_accuracy'], 'r--', color="blue")
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train ' + name1, 'Val ' + name1, 'Train ' + name2, 'Val ' + name2], loc='lower right')
    plt.show()

# Función para comparar la pérdida de dos entrenamientos
def plot_compare_losses(history1, history2, name1="Model 1", name2="Model 2", title="Comparison of Loss"):
    plt.plot(history1.history['loss'], color="green")
    plt.plot(history1.history['val_loss'], 'r--', color="green")
    plt.plot(history2.history['loss'], color="blue")
    plt.plot(history2.history['val_loss'], 'r--', color="blue")
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train ' + name1, 'Val ' + name1, 'Train ' + name2, 'Val ' + name2], loc='upper right')
    plt.show()

# Definir y compilar el modelo que usa activación sigmoid en las capas ocultas
model_sigmoid = Sequential([
    Dense(128, activation='sigmoid', input_shape=(784,)),  # Capa densa con 128 neuronas y activación sigmoid
    Dense(128, activation='sigmoid'),                      # Segunda capa densa con activación sigmoid
    Dense(128, activation='sigmoid'),                      # Tercera capa densa con activación sigmoid
    Dense(10, activation='softmax')                        # Capa de salida con activación softmax para clasificación
])

# Definir y compilar el modelo que usa activación ReLU en las capas ocultas
model_relu = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),    # Capa densa con 128 neuronas y activación ReLU
    Dense(128, activation='relu'),                        # Segunda capa densa con activación ReLU
    Dense(128, activation='relu'),                        # Tercera capa densa con activación ReLU
    Dense(10, activation='softmax')                       # Capa de salida con activación softmax para clasificación
])

model_sigmoid.compile(optimizer='adam',                   # Usar el optimizador Adam para el ajuste automático del aprendizaje
                      loss='sparse_categorical_crossentropy',  # Función de pérdida para clasificación de múltiples clases
                      metrics=['accuracy'])                # Seguimiento de la precisión del modelo durante el entrenamiento
model_relu.compile(optimizer='adam',                     # Usar el optimizador Adam para el ajuste automático del aprendizaje
                   loss='sparse_categorical_crossentropy',  # Función de pérdida para clasificación de múltiples clases
                   metrics=['accuracy'])                  # Seguimiento de la precisión del modelo durante el entrenamiento

# Entrenar ambos modelos y guardar el historial para análisis posterior
history_sigmoid = model_sigmoid.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=0)
history_relu = model_relu.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=0)

# Visualizar resultados
plot_compare_accs(history_sigmoid, history_relu, name1="Sigmoid", name2="ReLU", title="Accuracy Comparison: Sigmoid vs ReLU")
plot_compare_losses(history_sigmoid, history_relu, name1="Sigmoid", name2="ReLU", title="Loss Comparison: Sigmoid vs ReLU")