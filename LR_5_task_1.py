# Импортируем библиотеку numpy для работы с матрицами
import numpy as np

# Определяем функцию активации, например, сигмоиду
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Создаем класс Neuron для хранения весов и смещения
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    # Метод feedforward для вычисления выхода нейрона по входам
    def feedforward(self, inputs):
        # Умножаем входы на веса, прибавляем смещение и применяем функцию активации
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

# Создаем объект нейрона с заданными весами и смещением
weights = np.array([0, 1]) # w1 = 0, w2 = 1
bias = 4 # b = 4
n = Neuron(weights, bias)

# Проверяем выход нейрона на примере входов
x = np.array([2, 3]) # x1 = 2, x2 = 3
print(n.feedforward(x)) # 0.9990889488055994
