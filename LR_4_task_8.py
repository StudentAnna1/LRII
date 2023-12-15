# імпортувати бібліотеки
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# зчитати дані з файлу
data = pd.read_csv("data.txt", header=None)
print(data.shape) # перевірити розмір даних
print(data.head()) # перевірити формат даних

# виділити ознаки та мітки
X = data.iloc[:, :-1].values # всі стовпці, крім останнього
y = data.iloc[:, -1].values # останній стовпець

# створити словник для відповідності між мітками та маркерами
label_marker = {0: "o", 1: "s", 2: "^", 3: "v"}

# побудувати діаграму розсіювання для вхідних даних
plt.figure()
plt.title("Вхідні дані")
for label in np.unique(y):
    plt.scatter(X[y == label, 0], X[y == label, 1], marker=label_marker[label], label=label)
plt.xlabel("Ознака 1")
plt.ylabel("Ознака 2")
plt.legend()
plt.show()

# розділити дані на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# створити та навчити модель kNN
knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X_train, y_train)

# зробити прогноз для тестової вибірки
y_pred = knn.predict(X_test)

# порівняти прогнозовані та фактичні мітки
print("Прогнозовані мітки:", y_pred)
print("Фактичні мітки:", y_test)

# обчислити точність моделі
acc = accuracy_score(y_test, y_pred)
print("Точність моделі:", acc)

# визначити клас для тестової точки
test_point = np.array([[4.5, 3.2]])
test_label = knn.predict(test_point)
print("Клас для тестової точки:", test_label)
