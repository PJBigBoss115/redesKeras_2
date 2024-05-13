import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import GlorotUniform
from keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt

# Cargar y preprocesar datos
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

# Inicializador Glorot Uniform
initializer = GlorotUniform()

# Configuración de optimizadores
optimizers = {
    'SGD': SGD(learning_rate=0.01, momentum=0.9),
    'Adam': Adam(),
    'RMSprop': RMSprop()
}

histories = {}

# Crear, compilar y entrenar modelos para cada optimizador
for name, optimizer in optimizers.items():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(784,), kernel_initializer=initializer),
        Dense(128, activation='relu', kernel_initializer=initializer),
        Dense(128, activation='relu', kernel_initializer=initializer),
        Dense(10, activation='softmax', kernel_initializer=initializer)
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    histories[name] = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=0)

# Función para comparar resultados
def plot_compare_metrics(histories, metric):
    plt.figure(figsize=(10, 5))
    for name, history in histories.items():
        plt.plot(history.history[metric], label=f'{name} Train')
        plt.plot(history.history['val_' + metric], '--', label=f'{name} Validation')
    plt.title(f'Comparison of {metric.capitalize()} using Different Optimizers')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.show()

# Visualización de la precisión y la pérdida para cada optimizador
plot_compare_metrics(histories, 'accuracy')
plot_compare_metrics(histories, 'loss')