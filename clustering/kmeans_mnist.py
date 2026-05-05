import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tensorflow as tf

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

x = x_train[:2000]
y = y_train[:2000]

x_flat = x.reshape(x.shape[0], -1)

kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(x_flat)

plt.figure(figsize=(10,10))

for i in range(10):
    idx = np.where(clusters == i)[0][:10]
    for j, k in enumerate(idx):
        plt.subplot(10, 10, i*10 + j + 1)
        plt.imshow(x[k], cmap='gray')
        plt.axis('off')

plt.savefig("kmeans_result.png")
plt.close()

print("\n=== ANALISIS CLUSTER ===")

for i in range(10):
    labels = y[clusters == i]
    if len(labels) > 0:
        dominant = np.bincount(labels).argmax()
        print(f"Cluster {i} dominan angka: {dominant}")