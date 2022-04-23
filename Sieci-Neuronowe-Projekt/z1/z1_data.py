from math import exp
import numpy as np

learning_rate = 1
sigma = 1
x_plot = np.linspace(0, 1, 100)

x_input = np.array(
    [
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ]
)

y_and = np.array([[0], [0], [0], [1]])

y_xor = np.array([[1], [0], [0], [1]])

k_xor = [exp(-pow(s, 2) / (2 * sigma))
         for s in x_input[:, -2] - x_input[:, -1]]
x_kernelized = np.zeros((x_input.shape[0], x_input.shape[1] + 1))
x_kernelized[:, 0:3] = x_input
x_kernelized[:, -1:] = np.reshape(k_xor, (x_kernelized.shape[0], 1))

w = np.array([1/2, 0, 1])
w_kernelized = np.array([1/2, 0, 1, 0])
