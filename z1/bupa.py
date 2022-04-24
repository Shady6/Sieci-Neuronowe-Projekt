import matplotlib.pyplot as plt

from data import w, x_plot, x_input, y_and, y_xor, learning_rate, x_kernelized, w_kernelized
from lib import neuron_process, plot, plot3d, calc_y_plot


def calc_next_weights_bupa(w, x, y, learning_rate):
    z_sum = 0
    signs = []
    for i in range(len(x)):
        predicted = neuron_process(x[i], w)
        signs.append(y[i] - predicted)
        z_sum = z_sum + signs[-1] * x[i]
    z_sum = z_sum * learning_rate
    next_weights = w + z_sum
    all_guesses_correct = not sum(signs).sum()
    return (next_weights, all_guesses_correct)


def train(x, y, w, max_iterations=100, make_plot=True):
    i = 0
    curr_weights = w

    ys_plot = [calc_y_plot(curr_weights, x_plot)]

    while i < max_iterations:

        curr_weights, all_guesses_correct = \
            calc_next_weights_bupa(
                curr_weights, x, y, learning_rate)

        ys_plot.append(calc_y_plot(curr_weights, x_plot))

        i += 1
        if all_guesses_correct:
            break

    if make_plot:
        plot(x_input, x_plot, ys_plot, curr_weights, i, y)
    return curr_weights, i


plt.title("BUPA AND")
print('[BUPA] Training AND function')
train(x_input, y_and, w)

plt.title("BUPA XOR")
print('[BUPA] Training XOR function')
train(x_input, y_xor, w)

print('[BUPA] Training XOR function kernelized')
result_weights, iterations = train(
    x_kernelized, y_xor, w_kernelized, make_plot=False)
plot3d(x_kernelized, x_plot, result_weights,
       iterations, 'BUPA XOR Kernelized', y_xor)
