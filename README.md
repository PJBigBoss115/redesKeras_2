# redesKeras_2
En esta segunda parte, vamos a continuar desarrollando el problema de Fashion MNIST, con el objetivo de entender los aspectos prácticos del entrenamiento de redes neuronales.

## 1. Unidades de activación
En este ejercicio, vamos a evaluar la importancia de utilizar las unidades de activación adecuadas. Las funciones de activación como sigmoid han dejado de utilizarse en favor de otras unidades como ReLU.

**Ejercicio 1 *(2.5 puntos)***: Partiendo de una red sencilla como la desarrollada en el Trabajo 1, escribir un breve análisis comparando la utilización de unidades sigmoid y ReLU (por ejemplo, se pueden comentar aspectos como velocidad de convergencia, métricas obtenidas...). Explicar por qué pueden darse estas diferencias. Opcionalmente, comparar con otras activaciones disponibles en Keras.

*Pista: Usando redes más grandes se hace más sencillo apreciar las diferencias. Es mejor utilizar al menos 3 o 4 capas densas.*

### Solucion:
```python
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
```

#### ReLU vs. Sigmoid:
ReLU generalmente converge más rápido y proporciona mejores resultados en redes profundas debido a la menor probabilidad de que ocurra el problema de desvanecimiento de gradiente en comparación con la función sigmoid.
#### Visualización: 
Las gráficas mostrarán cómo los modelos con ReLU pueden comenzar a aprender más rápido y alcanzar una mayor precisión en comparación con sigmoid, que puede sufrir de saturación de neuronas especialmente en las capas más profundas.

## 2. Inicialización de parámetros

En este ejercicio, vamos a evaluar la importancia de una correcta inicialización de parámetros en una red neuronal.

**Ejercicio 2 *(2.5 puntos)***: Partiendo de una red similar a la del ejercicio anterior (usando ya ReLUs), comentar las diferencias que se aprecian en el entrenamiento al utilizar distintas estrategias de inicialización de parámetros. Para ello, inicializar todas las capas con las siguientes estrategias, disponibles en Keras, y analizar sus diferencias:

* Inicialización con ceros.
* Inicialización con una variable aleatoria normal.
* Inicialización con los valores por defecto de Keras para una capa Dense (estrategia *glorot uniform*)

### Solucion:
```python
# Inicializadores
zero_initializer = Zeros()
# Variable aleatoria normal
normal_initializer = RandomNormal()
# Inicializacion con los valores por defecto (Estrategia Glorot Uniform)
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
```

## 3. Optimizadores
**Ejercicio 3 *(2.5 puntos)***: Partiendo de una red similar a la del ejercicio anterior (utilizando la mejor estrategia de inicialización observada), comparar y analizar las diferencias que se observan  al entrenar con varios de los optimizadores vistos en clase, incluyendo SGD como optimizador básico (se puede explorar el espacio de hiperparámetros de cada optimizador, aunque para optimizadores más avanzados del estilo de adam y RMSprop es buena idea dejar los valores por defecto provistos por Keras).

### Solucion:
```python
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
```

## 4. Regularización y red final *(2.5 puntos)*
**Ejercicio 4.1**: Entrenar una red final que sea capaz de obtener una accuracy en el validation set cercana al 90%. Para ello, combinar todo lo aprendido anteriormente y utilizar técnicas de regularización para evitar overfitting. Algunos de los elementos que pueden tenerse en cuenta son los siguientes.

* Número de capas y neuronas por capa
* Optimizadores y sus parámetros
* Batch size
* Unidades de activación
* Uso de capas dropout, regularización L2, regularización L1...
* Early stopping (se puede aplicar como un callback de Keras, o se puede ver un poco "a ojo" cuándo el modelo empieza a caer en overfitting y seleccionar el número de epochs necesarias)
* Batch normalization

Si los modelos entrenados anteriormente ya se acercaban al valor requerido de accuracy, probar distintas estrategias igualmente y comentar los resultados.

Explicar brevemente la estrategia seguida y los modelos probados para obtener el modelo final, que debe verse entrenado en este Notebook. No es necesario guardar el entrenamiento de todos los modelos que se han probado, es suficiente con explicar cómo se ha llegado al modelo final.

### Solucion:
```python
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
```

## Estrategia para el Modelo Final
+ Número de capas y neuronas: Utilizar una estructura profunda con suficientes neuronas para capturar la complejidad del problema sin caer en overfitting.

+ Optimizadores: Basándonos en experimentos anteriores, emplearemos Adam por su eficiencia y adaptabilidad.

+ Batch size: Ajustar el tamaño del lote para equilibrar la velocidad de entrenamiento y la estabilidad del aprendizaje.

+ Unidades de Activación: ReLU se ha mostrado eficaz en capas ocultas por su capacidad para mitigar el problema del gradiente desvanecido.

+ Regularización: Incorporar Dropout para reducir el overfitting al añadir aleatoriedad durante el entrenamiento. Además, la Batch Normalization ayuda a normalizar las entradas de cada capa, acelerando la convergencia.

+ Early Stopping: Utilizar como un callback para detener el entrenamiento cuando el rendimiento en el conjunto de validación deje de mejorar, evitando así el overfitting.

+ Batch Normalization: Aplicarla después de cada capa densa para estabilizar el aprendizaje y mejorar el rendimiento.

#### Explicación del Modelo Final
Capas densas con más neuronas en la primera capa, decreciendo el número en capas subsiguientes para formar una estructura de embudo que ayuda a concentrar la información relevante.

Dropout y Batch Normalization después de cada capa para regularizar y mejorar la eficiencia del entrenamiento.

Early Stopping para cortar el entrenamiento si el modelo deja de mejorar, lo cual protege contra el overfitting y optimiza los recursos computacionales.

### Evaluación del modelo en datos de test
Una vez elegido el que creemos que es nuestro mejor modelo a partir de la estimación que hemos visto en los datos de validación, es hora de utilizar los datos de test para ver cómo se comporta nuestro modelo ante nuevos datos. Si hemos hecho bien las cosas, este número debería ser parecido al valor de nuestra estimación vista en los datos de validación.

**Pregunta 4.2**. Utilizando nuestro mejor modelo, obtener la accuracy resultante en el dataset de test. Comentar este resultado.

### Solucion:
```python
# Evaluación del modelo en el conjunto de test
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

# Impresión de la precisión obtenida en el conjunto de test
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Comparación con la precisión en validación
val_accuracy = max(history.history['val_accuracy'])  # Mejor precisión de validación durante el entrenamiento
print(f"Best Validation Accuracy: {val_accuracy * 100:.2f}%")
```

## Comentarios sobre los resultados
1. Precisión en el Test vs. Validación: El resultado de la precisión en el conjunto de test debería ser comparado con la mejor precisión observada en el conjunto de validación. Si ambos valores son cercanos, indica que el modelo ha generalizado bien y no ha sufrido de overfitting significativo.

2. Discrepancia entre Test y Validación: Una gran diferencia entre estas métricas podría sugerir problemas como:

    + Overfitting al conjunto de validación: Esto puede ocurrir si el conjunto de validación no es representativo del conjunto general de datos o si se ha realizado una optimización excesiva en las decisiones de modelado basadas en este conjunto.
    + Underfitting: Si ambos, la precisión del test y la validación son bajas, el modelo podría no ser suficientemente complejo o bien entrenado para capturar las regularidades en los datos.
3. Expectativas Realistas: Es importante recordar que la precisión en el mundo real puede ser menor que la observada durante el entrenamiento y las pruebas, especialmente si existen diferencias en la distribución de los datos entre el ambiente de entrenamiento/testeo y la aplicación real.
