# Імпорт необхідних бібліотек
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
# Завантаження даних з файлу
X = np.loadtxt('data_imbalance.txt', delimiter=',')
# Розділення даних на ознаки та мітки
X_train, y_train = X[:, :-1], X[:, -1]
# Побудова графіка вхідних даних
plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', cmap='rainbow')
plt.xlabel('Перша ознака')
plt.ylabel('Друга ознака')
plt.title('Вхідні дані')
plt.show()
# Створення об'єкта логістичної регресії
lr = LogisticRegression()
# Навчання моделі на даних
lr.fit(X_train, y_train)
# Отримання передбачень для даних
y_pred = lr.predict(X_train)
# Вивід точності та F1-міри моделі
print("Точність логістичної регресії =", accuracy_score(y_train, y_pred))
print("F1-міра логістичної регресії =", f1_score(y_train, y_pred))
# Побудова графіка границі класифікатора
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
plt.figure(figsize=(10, 6))
Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4, cmap='rainbow')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', cmap='rainbow')
plt.xlabel('Перша ознака')
plt.ylabel('Друга ознака')
plt.title('Логістична регресія')
plt.show()
# Створення об'єктів оверсемплінгу та андерсемплінгу
smote = SMOTE()
rus = RandomUnderSampler()
# Застосування оверсемплінгу та андерсемплінгу до даних
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
# Побудова графіків даних після оверсемплінгу та андерсемплінгу
plt.figure(figsize=(10, 6))
plt.scatter(X_train_smote[:, 0], X_train_smote[:, 1], c=y_train_smote, marker='o', cmap='rainbow')
plt.xlabel('Перша ознака')
plt.ylabel('Друга ознака')
plt.title('Дані після оверсемплінгу')
plt.show()
plt.figure(figsize=(10, 6))
plt.scatter(X_train_rus[:, 0], X_train_rus[:, 1], c=y_train_rus, marker='o', cmap='rainbow')
plt.xlabel('Перша ознака')
plt.ylabel('Друга ознака')
plt.title('Дані після андерсемплінгу')
plt.show()
# Навчання моделі на даних після оверсемплінгу
lr.fit(X_train_smote, y_train_smote)
# Отримання передбачень для даних після оверсемплінгу
y_pred_smote = lr.predict(X_train_smote)
class_weights = {0: 1, 1: 10}
# Створення об'єкта випадкового лісу з вагами класів
rf = RandomForestClassifier(n_estimators=100, random_state=0, class_weight=class_weights)
# Навчання моделі на даних
rf.fit(X_train, y_train)
# Отримання передбачень для даних
y_pred = rf.predict(X_train)
# Вивід точності та F1-м
print("Точність випадкового лісу з вагами класів =", accuracy_score(y_train, y_pred))
print("F1-міра випадкового лісу з вагами класів =", f1_score(y_train, y_pred))
# Побудова графіка границі класифікатора
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
plt.figure(figsize=(10, 6))
Z = rf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4, cmap='rainbow')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', cmap='rainbow')
plt.xlabel('Перша ознака')
plt.ylabel('Друга ознака')
plt.title('Випадковий ліс з вагами класів')
plt.show()
