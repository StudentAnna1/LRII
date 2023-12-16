# імпортувати бібліотеки
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# зчитати дані з файлу
data = pd.read_excel("Відстані між обласними центрами України.xlsx", header=[0,1])
print(data.shape) # перевірити розмір даних
print(data.head()) # перевірити формат даних

# виділити матрицю відстаней
D = data.iloc[1:, 2:].values # пропустити перший рядок і стовпець
D = D.astype(float) # перетворити на числа з плаваючою комою
print(D) # вивести матрицю відстаней

# визначити параметри ACO
n = 50 # кількість мурах
m = 100 # кількість ітерацій
rho = 0.1 # коефіцієнт випаровування
alpha = 1 # коефіцієнт інтенсивності
beta = 2 # коефіцієнт евристики

# ініціалізувати матрицю феромонів
P = np.ones((24, 24)) * 0.01 # невеликі додатні значення
np.fill_diagonal(P, 0) # нулі на діагоналі
print(P) # вивести матрицю феромонів

# почати головний цикл ACO
best_solution = None # змінна для зберігання найкращого рішення
best_distance = np.inf # змінна для зберігання найкращої відстані
for i in range(m): # повторити m разів
    print(f"Iteration {i+1}") # вивести номер ітерації
    solutions = [] # список для зберігання рішень для кожної мурахи
    for j in range(n): # для кожної мурахи
        # вибрати Вінницю (номер 24) як початкову точку
        start = 23 # індекс початкової точки
        visited = [start] # список відвіданих міст
        distance = 0 # загальна відстань
        # поки не відвідано усі міста
        while len(visited) < 24:
            # знайти список можливих міст, які ще не відвідано
            possible = list(set(range(24)) - set(visited))
            # обчислити ймовірності переходу до кожного можливого міста
            probabilities = []
            for city in possible:
                # використати формулу, що враховує рівень феромону та евристичну інформацію (обернена відстань)
                p = (P[visited[-1], city] ** alpha) * ((1 / D[visited[-1], city]) ** beta)
                probabilities.append(p)
            # нормалізувати ймовірності, щоб їх сума дорівнювала 1
            probabilities = probabilities / np.sum(probabilities)
            # вибрати наступне місто за допомогою рулеткового відбору
            next_city = random.choices(possible, weights=probabilities)[0]
            # додати наступне місто до списку відвіданих міст та оновити загальну відстань
            visited.append(next_city)
            distance += D[visited[-2], next_city]
        # додати початкову точку до кінця списку відвіданих міст та оновити загальну відстань
        visited.append(start)
        distance += D[visited[-2], start]
        # зберегти список відвіданих міст та загальну відстань для кожної мурахи у вигляді кортежу
        solutions.append((visited, distance))
        # знайти найкраще рішення серед усіх мурах
    best_solution = min(solutions, key=lambda x: x[1])
    # оновити найкращу відстань, якщо потрібно
    if best_solution[1] < best_distance:
        best_distance = best_solution[1]
    # вивести найкраще рішення та його відстань
    print("Найкраще рішення:", best_solution[0])
    print("Найкраща відстань:", best_solution[1])
    # оновити матрицю феромонів
    for k in range(n):  # для кожної мурахи
        # взяти її рішення та відстань
        solution, distance = solutions[k]
        # для кожної пари міст у рішенні
        for l in range(len(solution) - 1):
            # використати формулу, що враховує випаровування та накопичення феромону
            P[solution[l], solution[l + 1]] = (1 - rho) * P[solution[l], solution[l + 1]] + (1 / distance)
# закінчити головний цикл ACO
# вивести найкраще рішення та його відстань
print("Найкраще рішення за усі ітерації:", best_solution[0])
print("Найкраща відстань за усі ітерації:", best_distance)
# побудувати графік найкоротшого маршруту
plt.figure()
plt.title("Найкоротший маршрут")
# використати координати міст з файлу data.txt
coordinates = data.iloc[1:, 0].values  # перший стовпець

coordinates = [eval(x) for x in coordinates]  # перетворити на кортежі
coordinates = np.array(coordinates)  # перетворити на масив numpy
# малювати точки міст
plt.scatter(coordinates[:, 0], coordinates[:, 1], color="blue")
# малювати лінії між містами за найкращим рішенням
for i in range(len(best_solution[0]) - 1):
    plt.plot(coordinates[best_solution[0][i:i + 2], 0], coordinates[best_solution[0][i:i + 2], 1], color="red")
# показати графік
plt.show()
