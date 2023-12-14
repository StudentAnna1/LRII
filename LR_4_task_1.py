# Імпорт необхідних бібліотек
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
# Завантаження даних з файлу
X = np.loadtxt('data_random_forests.txt', delimiter=',')
# Розділення даних на ознаки та мітки
X_train, y_train = X[:, :-1], X[:, -1]
# Створення об'єктів випадкового та гранично випадкового лісів
rf = RandomForestClassifier(n_estimators=100, random_state=0)
et = ExtraTreesClassifier(n_estimators=100, random_state=0)
# Навчання моделей на даних
rf.fit(X_train, y_train)
et.fit(X_train, y_train)
# Отримання передбачень для даних
y_pred_rf = rf.predict(X_train)
y_pred_et = et.predict(X_train)
# Вивід точності моделей
print("Точність випадкового лісу =", accuracy_score(y_train, y_pred_rf))
print("Точність гранично випадкового лісу =", accuracy_score(y_train, y_pred_et))
# Побудова графіка вхідних даних
plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', cmap='rainbow')
plt.xlabel('Перша ознака')
plt.ylabel('Друга ознака')
plt.title('Вхідні дані')
plt.show()
# Побудова графіків границь класифікаторів
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
# Випадковий ліс
plt.figure(figsize=(10, 6))
Z = rf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4, cmap='rainbow')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', cmap='rainbow')
plt.xlabel('Перша ознака')
plt.ylabel('Друга ознака')
plt.title('Випадковий ліс')
plt.show()
# Гранично випадковий ліс
plt.figure(figsize=(10, 6))
Z = et.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4, cmap='rainbow')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', cmap='rainbow')
plt.xlabel('Перша ознака')
plt.ylabel('Друга ознака')
plt.title('Гранично випадковий ліс')
plt.show()
