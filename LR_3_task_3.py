import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Вхідний файл, який містить дані
input_file = 'data_multivar_regr.txt'
# Завантаження даних
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]
# Створення об'єкта лінійного регресора
regressor = LinearRegression()
# Навчання моделі на даних
regressor.fit(X, y)
# Прогнозування результату
y_pred = regressor.predict(X)
# Побудова графіка
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, color='green')
ax.plot(X[:, 0], X[:, 1], y_pred, color='black', linewidth=4)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')
plt.show()
# Виведення коефіцієнтів регресії
print("Slope =", regressor.coef_)
print("Intercept =", regressor.intercept_)
# Виведення показників якості моделі
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, explained_variance_score, r2_score
print("Mean absolute error =", mean_absolute_error(y, y_pred))
print("Mean squared error =", mean_squared_error(y, y_pred))
print("Median absolute error =", median_absolute_error(y, y_pred))
print("Explained variance score =", explained_variance_score(y, y_pred))
print("R2 score =", r2_score(y, y_pred))
