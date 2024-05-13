import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.initializers import GlorotUniform
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# Cargar el dataset Fashion MNIST, que contiene imágenes de ropa y accesorios.
# Las imágenes se normalizan dividiendo por 255 para transformar los píxeles en valores entre 0 y 1.
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0  # 784 es el número de píxeles por imagen (28x28).
x_test = x_test.reshape(-1, 784) / 255.0

# Inicialización del modelo utilizando la API Sequential de Keras.
model = Sequential([
    # Capa densa con 512 neuronas y función de activación ReLU. 
    # Utiliza la inicialización GlorotUniform, adecuada para mantener la varianza de los outputs.
    Dense(512, activation='relu', input_shape=(784,), kernel_initializer=GlorotUniform()),
    BatchNormalization(),  # Normaliza la activación de la capa anterior, estabilizando el aprendizaje.
    Dropout(0.3),  # Aplica dropout al 30% de las neuronas, para reducir el overfitting.
    Dense(256, activation='relu', kernel_initializer=GlorotUniform()),  # Segunda capa densa con 256 neuronas.
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu', kernel_initializer=GlorotUniform()),  # Tercera capa densa con 128 neuronas.
    BatchNormalization(),
    Dropout(0.3),
    Dense(10, activation='softmax', kernel_initializer=GlorotUniform())  # Capa de salida con 10 clases.
])

# Compilación del modelo especificando el optimizador Adam, la función de pérdida y la métrica de precisión.
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Configuración de EarlyStopping para monitorear la precisión de validación y detener el entrenamiento
# si no hay mejora después de 5 épocas, restaurando los mejores pesos encontrados.
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Entrenamiento del modelo con datos de entrenamiento y validación, utilizando el callback de EarlyStopping.
history = model.fit(
    x_train, y_train,
    epochs=50,  # Número máximo de épocas.
    batch_size=64,  # Tamaño del lote.
    validation_data=(x_test, y_test),
    callbacks=[early_stopping],  # Lista de callbacks.
    verbose=2  # Modo de verbosidad 2, que muestra una línea por época.
)

# Visualización del desempeño del modelo.
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Performance')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#######################################################################################

# Evaluación del modelo en el conjunto de test
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

# Impresión de la precisión obtenida en el conjunto de test
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Comparación con la precisión en validación
val_accuracy = max(history.history['val_accuracy'])  # Mejor precisión de validación durante el entrenamiento
print(f"Best Validation Accuracy: {val_accuracy * 100:.2f}%")