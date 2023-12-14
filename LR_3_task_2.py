import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# Вхідний файл, який містить дані
input_file = 'data_regr_1.txt'
# Завантаження даних
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, 0], data[:, 1]
# Перетворення X в двовимірний масив
X = X.reshape(-1, 1)
# Створення об'єкта лінійного регресора
regressor = LinearRegression()
# Навчання моделі на даних
regressor.fit(X, y)
# Прогнозування результату
y_pred = regressor.predict(X)
# Побудова графіка
plt.scatter(X, y, color='green')
plt.plot(X, y_pred, color='black', linewidth=4)
plt.xlabel('X')
plt.ylabel('y')
plt.show()
# Виведення коефіцієнтів регресії
print("Slope =", regressor.coef_[0])
print("Intercept =", regressor.intercept_)
# Виведення показників якості моделі
print("R-squared =", regressor.score(X, y))
