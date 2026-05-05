import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# ======================
# LOAD DATA
# ======================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

# ======================
# MODEL 1: MLP
# ======================
model_mlp = models.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model_mlp.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

history_mlp = model_mlp.fit(x_train, y_train, epochs=5)

# ======================
# MODEL 2: CNN
# ======================
x_train_cnn = x_train[..., np.newaxis]
x_test_cnn = x_test[..., np.newaxis]

model_cnn = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model_cnn.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

history_cnn = model_cnn.fit(x_train_cnn, y_train, epochs=5)

# ======================
# MODEL 3: CNN DEEP
# ======================
model_cnn2 = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model_cnn2.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

history_cnn2 = model_cnn2.fit(x_train_cnn, y_train, epochs=5)

# ======================
# EVALUASI
# ======================
mlp_loss, mlp_acc = model_mlp.evaluate(x_test, y_test)
cnn_loss, cnn_acc = model_cnn.evaluate(x_test_cnn, y_test)
cnn2_loss, cnn2_acc = model_cnn2.evaluate(x_test_cnn, y_test)

print("\n=== HASIL AKURASI ===")
print("MLP  :", mlp_acc)
print("CNN  :", cnn_acc)
print("CNN2 :", cnn2_acc)

# ======================
# GRAFIK AKURASI
# ======================
plt.plot(history_mlp.history['accuracy'], label='MLP')
plt.plot(history_cnn.history['accuracy'], label='CNN')
plt.plot(history_cnn2.history['accuracy'], label='CNN2')
plt.title("Perbandingan Akurasi")
plt.legend()
plt.savefig("accuracy.png")
plt.close()

# ======================
# GRAFIK LOSS
# ======================
plt.plot(history_mlp.history['loss'], label='MLP')
plt.plot(history_cnn.history['loss'], label='CNN')
plt.plot(history_cnn2.history['loss'], label='CNN2')
plt.title("Perbandingan Loss")
plt.legend()
plt.savefig("loss.png")
plt.close()

# ======================
# CONTOH PREDIKSI
# ======================
pred = model_cnn.predict(x_test_cnn)
pred_classes = np.argmax(pred, axis=1)

for i in range(5):
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Pred: {pred_classes[i]} | True: {y_test[i]}")
    plt.savefig(f"pred_{i}.png")
    plt.close()