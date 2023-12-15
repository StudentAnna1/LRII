import numpy as np # Імпортуємо бібліотеку numpy для роботи з масивами
from matplotlib import pyplot as plt # Імпортуємо бібліотеку matplotlib для візуалізації даних
from sklearn.neighbors import NearestNeighbors # Імпортуємо клас NearestNeighbors з бібліотеки sklearn для пошуку найближчих сусідів

X = np.array(
    [[2.1, 1.3], [1.3, 3.2], [2.9, 2.5], [2.7, 5.4], [3.8, 5.4], [3.8, 0.9], [7.3, 2.1], [4.2, 6.5], [3.8, 3.7],
     [2.5, 4.1], [3.4, 1.9], [5.7, 3.5], [6.1, 4.3], [5.1, 2.2], [6.2, 1.1]]) # Створюємо масив X з вхідними даними
k = 5 # Встановлюємо кількість найближчих сусідів, яких хочемо знайти
test_datapoint = [4.3, 2.7] # Створюємо масив test_datapoint з тестовою точкою
# Змінюємо форму test_datapoint до двовимірного масиву
test_datapoint = np.reshape(test_datapoint, (1, -1))
plt.figure() # Створюємо нове вікно для графіка
plt.title('Вхідні дані') # Додаємо заголовок до графіка
plt.scatter(X[:, 0], X[:, 1], marker='o', s=75, color='black') # Малюємо точки вхідних даних чорним кольором
knn_model = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X) # Створюємо модель knn_model з алгоритмом ball_tree і навчаємо її на вхідних даних X
distances, indices = knn_model.kneighbors(test_datapoint) # Знаходимо відстані і індекси k найближчих сусідів для тестової точки
print("\nK Nearest Neighboars:") # Виводимо повідомлення про k найближчих сусідів
for rank, index in enumerate(indices[0][:k], start=1): # Проходимо по індексах k найближчих сусідів
    print(str(rank) + "==>", X[index]) # Виводимо ранг і координати сусіда
plt.figure() # Створюємо нове вікно для графіка
plt.title('Ближайшие соседи') # Додаємо заголовок до графіка
plt.scatter(X[:, 0], X[:, 1], marker='o', s=75, color='k') # Малюємо точки вхідних даних чорним кольором
plt.scatter(X[indices][0][:][:, 0], X[indices][0][:][:, 1], marker='o', s=250, color='k', facecolors='none') # Малюємо кільця навколо k найближчих сусідів
plt.scatter(test_datapoint[0][0], test_datapoint[0][1], marker='x', s=75, color='k') # Малюємо тестову точку хрестиком
plt.show() # Показуємо графік
