# Імпорт необхідних бібліотек
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Вхідний файл, який містить дані
input_file = 'data_clustering.txt'
# Завантаження даних
X = np.loadtxt(input_file, delimiter=',')
# Створення об'єкта k-середніх з 5 кластерами
kmeans = KMeans(n_clusters=5)
# Навчання моделі на даних
kmeans.fit(X)
# Отримання міток кластерів для даних
y = kmeans.labels_
# Отримання координат центрів кластерів
centroids = kmeans.cluster_centers_
# Побудова графіка
plt.scatter(X[:, 0], X[:, 1], c=y, marker='o')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Кластеризація даних методом k-середніх')
plt.show()
