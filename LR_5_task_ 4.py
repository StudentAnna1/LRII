import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

tex = np.loadtxt('data_simple_nn.txt')

data = tex[:, :2]
labels = tex[:, 2:]
plt.figure()
plt.scatter(data[:,0],data[:,1])
plt.xlabel('Razmernost 1')
plt.ylabel('Razmernost 2')
plt.title('Input data')
dim1_min, dim1_max = data[:,0].min(),data[:,0].max()
dim2_min, dim2_max = data[:,1].min(),data[:,1].max()
num_output = labels.shape[1]
dim1 = [dim1_min,dim1_max]
dim2 = [dim2_min, dim2_max]
nn = nl.net.newp([dim1,dim2], num_output)
error_progress = nn.train(data, labels, epochs=100,show=20, lr=0.03)
plt.figure()
plt.plot(error_progress)
plt.xlabel('Количество эпох')
plt.ylabel('Ошибка обучения')
plt.title('Изменение ошибки обучения')
plt.grid()
plt.show()
print('\nTest results:')
data_test = [[0.4,4.3],[4.4,0.6],[4.7,8.1]]
for item in data_test:
    print(item, '-->', nn.sim([item])[0])
