from matplotlib import pyplot as plt
import numpy as np

from plot import plot_cost_changes, plot_weight_changes
from data import sigma, eps, w1, w2, x, y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def calculate_output(x, w1, w2):
    z1 = np.dot(w1, x)
    a1 = sigmoid(z1)
    a1 = np.insert(a1, 0, 1)
    z2 = np.dot(w2, a1)
    a2 = sigmoid(z2)
    return z1, z2, a1, a2


def backprop(w1, w2, z1, z2, a1, a2, x, y):
    dc_doutput = 2 * (a2 - y)
    doutput_dz2 = sigmoid_derivative(z2)
    dz2 = dc_doutput * doutput_dz2

    dw2 = dz2 * a1

    dw1_n1 = dz2 * w2[1] * sigmoid_derivative(z1[0]) * x
    dw1_n2 = dz2 * w2[2] * sigmoid_derivative(z1[1]) * x
    dw1 = np.array([dw1_n1, dw1_n2])
    return dw1, dw2


def train_partial(w1, w2, max_iter=1000, break_on_eps=False):
    w1s = [w1]
    w2s = [w2]
    costs = []
    dw1s = []
    dw2s = []

    breakpoints = [0]
    for x_train, y_train in zip(x, y):
        i = 0
        while i < max_iter:
            z1, z2, a1, a2 = calculate_output(x_train, w1, w2)
            cost = (y_train - a2)**2
            dw1, dw2 = backprop(w1, w2, z1, z2, a1, a2, x_train, y_train)

            if break_on_eps and np.abs(dw1).sum() + np.abs(dw2).sum() < eps:
                break

            w1 = w1 - sigma * dw1
            w2 = w2 - sigma * dw2

            w1s.append(w1)
            w2s.append(w2)
            dw1s.append(dw1)
            dw2s.append(dw2)
            costs.append(cost)
            i += 1

        breakpoints.append(breakpoints[-1] + i)

    plot_weight_changes(dw1s, dw2s,
                        breakpoints[1:], title='Zmiana energii w trybie cząstkowym', gradientPlot=True)
    plot_weight_changes(w1s, w2s,
                        breakpoints[1:], title='Zmiany wag w trybie cząstkowym')
    plot_cost_changes(costs,
                      breakpoints[1:], title='Zmiana kosztu w trybie cząstkowym')
    plt.show()

    return w1, w2


def train_complete(w1, w2, max_iter=10000, break_on_eps=False):
    w1s = [w1]
    w2s = [w2]
    costs = []
    dw1s = []
    dw2s = []

    i = 0
    while i < max_iter:
        single_train_costs = []
        single_train_dw1s = []
        single_train_dw2s = []
        for x_train, y_train in zip(x, y):
            z1, z2, a1, a2 = calculate_output(x_train, w1, w2)
            cost = (y_train - a2)**2
            dw1, dw2 = backprop(w1, w2, z1, z2, a1, a2, x_train, y_train)

            single_train_costs.append(cost)
            single_train_dw1s.append(dw1)
            single_train_dw2s.append(dw2)

        cost = sum(single_train_costs)
        dw1 = sum(single_train_dw1s)
        dw2 = sum(single_train_dw2s)

        if break_on_eps and np.abs(dw1).sum() + np.abs(dw2).sum() < eps:
            break

        w1 = w1 - sigma * dw1
        w2 = w2 - sigma * dw2

        w1s.append(w1)
        w2s.append(w2)
        dw1s.append(dw1)
        dw2s.append(dw2)
        costs.append(cost)
        i += 1

    plot_weight_changes(dw1s, dw2s,
                        title='Zmiana energii w trybie całkowitym', gradientPlot=True)
    plot_weight_changes(w1s, w2s,
                        title='Zmiany wag w trybie całkowitym')
    plot_cost_changes(costs,
                      title='Zmiana kosztu w trybie całkowitym')
    plt.show()

    return w1, w2


new_w1, new_w2 = train_partial(w1, w2, break_on_eps=False)
print('Tryb energii cząstkowej')
for x_train, y_train in zip(x, y):
    _, _, _, output = calculate_output(x_train, new_w1, new_w2)
    print(f'Given x: {x_train}\t Output: {output}')

new_w1, new_w2 = train_complete(w1, w2, break_on_eps=True, max_iter=11000)
print('Tryb energii całkowitej')
for x_train, y_train in zip(x, y):
    _, _, _, output = calculate_output(x_train, new_w1, new_w2)
    print(f'Given x: {x_train}\t Output: {output}')
