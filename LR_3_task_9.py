# Імпорт необхідних бібліотек
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
# Завантаження даних з файлу
X = np.loadtxt('data_clustering.txt', delimiter=',')
# Створення об'єкта зсуву середньою з пропускною здатністю 2
ms = MeanShift(bandwidth=2)
# Навчання моделі на даних
ms.fit(X)
# Отримання міток кластерів для даних
y_pred = ms.labels_
# Отримання координат центрів кластерів
centroids = ms.cluster_centers_
# Вивід кількості кластерів
print("Кількість кластерів =", len(centroids))
# Побудова графіка
plt.scatter(X[:, 0], X[:, 1], c=y_pred, marker='o')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
plt.xlabel('Перший атрибут')
plt.ylabel('Другий атрибут')
plt.title('Кластеризація набору даних зсувом середньою')
plt.show()
