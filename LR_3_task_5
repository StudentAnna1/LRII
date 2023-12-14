# Импорт необходимых библиотек
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Задание параметров
m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

# Построение графика случайных данных
plt.scatter(X, y, color='green')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Случайные данные')
plt.show()

# Создание объекта линейного регрессора
lin_reg = LinearRegression()
# Обучение модели на данных
lin_reg.fit(X, y)
# Прогнозирование результатов
y_pred_lin = lin_reg.predict(X)
# Вывод коэффициентов регрессии
print("Линейная регрессия:")
print("Slope =", lin_reg.coef_[0][0])
print("Intercept =", lin_reg.intercept_[0])
# Вывод показателей качества модели
print("Mean absolute error =", mean_absolute_error(y, y_pred_lin))
print("Mean squared error =", mean_squared_error(y, y_pred_lin))
print("R2 score =", r2_score(y, y_pred_lin))
# Построение графика линейной регрессии
plt.scatter(X, y, color='green')
plt.plot(X, y_pred_lin, color='black', linewidth=4)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Линейная регрессия')
plt.show()

# Создание объекта полиномиальных признаков
poly = PolynomialFeatures(degree=2)
# Преобразование данных в полиномиальную форму
X_poly = poly.fit_transform(X)
# Создание объекта линейного регрессора
poly_reg = LinearRegression()
# Обучение модели на полиномиальных данных
poly_reg.fit(X_poly, y)
# Прогнозирование результатов
y_pred_poly = poly_reg.predict(X_poly)
# Вывод коэффициентов регрессии
print("Полиномиальная регрессия:")
print("Slope =", poly_reg.coef_[0][1:])
print("Intercept =", poly_reg.intercept_[0])
# Вывод показателей качества модели
print("Mean absolute error =", mean_absolute_error(y, y_pred_poly))
print("Mean squared error =", mean_squared_error(y, y_pred_poly))
print("R2 score =", r2_score(y, y_pred_poly))
# Построение графика полиномиальной регрессии
plt.scatter(X, y, color='green')
plt.plot(X, y_pred_poly, color='black', linewidth=4)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Полиномиальная регрессия')
plt.show()
