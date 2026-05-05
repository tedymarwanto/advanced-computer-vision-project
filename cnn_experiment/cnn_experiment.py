import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ======================
# LOAD DATA (Fashion MNIST)
# ======================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train[..., None]
x_test = x_test[..., None]

# ======================
# FUNCTION MODEL
# ======================
def create_model(filters=32, layers_count=1):
    model = models.Sequential()

    model.add(layers.Conv2D(filters, (3,3), activation='relu', input_shape=(28,28,1)))

    for _ in range(layers_count-1):
        model.add(layers.Conv2D(filters, (3,3), activation='relu'))

    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# ======================
# EKSPERIMEN
# ======================
configs = [
    ("Model A (32 filter, 1 layer)", 32, 1),
    ("Model B (64 filter, 1 layer)", 64, 1),
    ("Model C (64 filter, 2 layer)", 64, 2),
]

results = []

for name, f, l in configs:
    print(f"\nTraining {name}")
    model = create_model(filters=f, layers_count=l)
    history = model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
    
    loss, acc = model.evaluate(x_test, y_test)
    print(name, "Accuracy:", acc)
    
    results.append((name, acc))

# ======================
# PRINT HASIL
# ======================
print("\n=== HASIL AKHIR ===")
for r in results:
    print(r[0], ":", r[1])