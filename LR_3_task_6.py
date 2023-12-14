# Импорт необходимых библиотек
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, learning_curve

# Задание параметров
m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Создание объекта линейного регрессора
lin_reg = LinearRegression()

# Построение кривой обучения для линейной регрессии
train_sizes, train_scores, test_scores = learning_curve(lin_reg, X_train, y_train, scoring='neg_mean_squared_error')
# Усреднение значений качества по фолдам
train_scores_mean = -train_scores.mean(axis=1)
test_scores_mean = -test_scores.mean(axis=1)
# Построение графика
plt.plot(train_sizes, train_scores_mean, label='Training error')
plt.plot(train_sizes, test_scores_mean, label='Test error')
plt.xlabel('Training set size')
plt.ylabel('MSE')
plt.title('Learning curve for linear regression')
plt.legend()
plt.show()

# Создание объекта полиномиальных признаков
poly = PolynomialFeatures(degree=2)
# Преобразование данных в полиномиальную форму
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
# Создание объекта линейного регрессора
poly_reg = LinearRegression()

# Построение кривой обучения для полиномиальной регрессии
train_sizes, train_scores, test_scores = learning_curve(poly_reg, X_train_poly, y_train, scoring='neg_mean_squared_error')
# Усреднение значений качества по фолдам
train_scores_mean = -train_scores.mean(axis=1)
test_scores_mean = -test_scores.mean(axis=1)
# Построение графика
plt.plot(train_sizes, train_scores_mean, label='Training error')
plt.plot(train_sizes, test_scores_mean, label='Test error')
plt.xlabel('Training set size')
plt.ylabel('MSE')
plt.title('Learning curve for polynomial regression')
plt.legend()
plt.show()
