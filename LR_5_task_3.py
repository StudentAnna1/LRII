import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

tex = np.loadtxt('data_perceptron.txt')

data = tex[:, :2]
labels = tex[:, 2].reshape((tex.shape[0],1))
plt.figure()
plt.scatter(data[:,0],data[:,1])
plt.xlabel('Razmernost 1')
plt.ylabel('Razmernost 2')
plt.title('Input data')
dim1_min, dim1_max, dim2_min, dim2_max = 0, 1, 0, 1
num_output = labels.shape[1]
dim1 = [dim1_min,dim1_max]
dim2 = [dim2_min, dim2_max]
perceptron = nl.net.newp([dim1,dim2], num_output)
error_progress = perceptron.train(data, labels, epochs=100,show=20, lr=0.03)
plt.figure()
plt.plot(error_progress)
plt.xlabel('Количество эпох')
plt.ylabel('Ошибка обучения')
plt.title('Изменение ошибки обучения')
plt.grid()
plt.show()

