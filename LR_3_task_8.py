# Імпорт необхідних бібліотек
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
# Завантаження набору даних Iris
X, y = load_iris(return_X_y=True)
# Створення об'єкта K-середніх з 3 кластерами
kmeans = KMeans(n_clusters=3)
# Навчання моделі на даних
kmeans.fit(X)
# Отримання міток кластерів для даних
y_pred = kmeans.labels_
# Отримання координат центрів кластерів
centroids = kmeans.cluster_centers_
# Побудова графіка
plt.scatter(X[:, 0], X[:, 1], c=y_pred, marker='o')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
plt.xlabel('Довжина чашолистка')
plt.ylabel('Ширина чашолистка')
plt.title('Кластеризація набору даних Iris методом K-середніх')
plt.show()
