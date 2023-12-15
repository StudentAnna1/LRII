# Импортируем библиотеку numpy для работы с матрицами
import numpy as np

# Определяем функцию активации, например, сигмоиду
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
class Neuron:
    def __init__(self,weights,bias):
        self.weights=weights
        self.bias=bias

    def feedforward(self, inputs):
        total = np.dot(self.weights,inputs) + self.bias
        return sigmoid(total)
class MatsukNeuralNetwork:
    def __init__(self):
        weights = np.array([0, 1])
        bias = 0
        self.h1 = Neuron(weights,bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    # Метод feedforward для вычисления выхода нейрона по входам
    def feedforward(self, inputs):
        # Умножаем входы на веса, прибавляем смещение и применяем функцию активации
        out_h1 = self.h1.feedforward(inputs)
        out_h2 = self.h2.feedforward(inputs)
        out_o1 = self.o1.feedforward(np.array([out_h1,out_h2]))
        return out_o1


network=MatsukNeuralNetwork()
X = np.array([2,3])
print(network.feedforward(X))



