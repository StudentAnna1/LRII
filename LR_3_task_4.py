# Импорт необходимых библиотек
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Загрузка данных
X, y = load_diabetes(return_X_y=True)
# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
# Создание объекта линейного регрессора
regr = LinearRegression()
# Обучение модели на обучающей выборке
regr.fit(X_train, y_train)
# Прогнозирование результатов на тестовой выборке
y_pred = regr.predict(X_test)
# Вывод коэффициентов регрессии
print("regr.coef_ =", regr.coef_)
print("regr.intercept_ =", regr.intercept_)
# Вывод показателей качества модели
print("r2_score =", r2_score(y_test, y_pred))
print("mean_absolute_error =", mean_absolute_error(y_test, y_pred))
print("mean_squared_error =", mean_squared_error(y_test, y_pred))
# Построение графика
plt.scatter(y_test, y_pred, color='green')
plt.plot([0, 350], [0, 350], color='black', linewidth=4)
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.show()
