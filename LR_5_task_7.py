# Импортируем библиотеки
import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl
from sklearn import datasets

# Загружаем данные об ирисах
iris = datasets.load_iris()
data = iris.data[:, :2] # Берем только два признака: длину и ширину чашелистика
labels = iris.target # Берем метки классов

# Нормализуем данные в диапазоне [0, 1]
data = (data - data.min()) / (data.max() - data.min())

# Определяем размерность входного и выходного слоев
input_dim = data.shape[1]
output_dim = 4 # Количество узлов на выходном слое

# Создаем карту Кохонена с размером сетки 2x2
net = nl.net.newc([[0, 1] for _ in range(input_dim)], output_dim)

# Обучаем карту Кохонена на данных
error = net.train(data, epochs=100, show=10)

# Визуализируем результаты
plt.figure()
plt.plot(error)
plt.xlabel('Epoch number')
plt.ylabel('Error')
plt.title('Training error')

plt.figure()
for i in range(output_dim):
    # Получаем веса узлов
    node_weights = net.layers[0].np['w'][i, :]
    # Находим ближайшие точки к узлу
    node_data = data[np.linalg.norm(data - node_weights, axis=1) < 0.2]
    # Рисуем точки и узел
    plt.scatter(node_data[:, 0], node_data[:, 1])
    plt.scatter(node_weights[0], node_weights[1], marker='*', s=200, c='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Kohonen map')
plt.show()
