import matplotlib.pyplot as plt

from z1_data import w, x_plot, x_input, y_and, y_xor, learning_rate
from z1_lib import neuron_process, plot, calc_y_plot


def calc_next_weights_pa(w, x,  expected, predicted, learning_rate):
    return w + learning_rate * (expected - predicted) * x


def train(x, y, max_iterations=100):
    i = 0
    curr_weights = w
    good_guess_times = 0

    ys_plot = [calc_y_plot(curr_weights, x_plot)]

    while good_guess_times != len(x) and i < max_iterations:
        j = i % len(x)
        predicted = neuron_process(x[j], curr_weights)

        if predicted == y[j]:
            good_guess_times += 1
            if good_guess_times == len(x):
                break
        else:
            good_guess_times = 0

        curr_weights = calc_next_weights_pa(
            curr_weights, x[j], y[j],  predicted, learning_rate)

        ys_plot.append(calc_y_plot(curr_weights, x_plot))

        i += 1

    plot(x_input, x_plot, ys_plot, curr_weights, i)


plt.title("PA AND")
print('[PA] Training AND function')
train(x_input, y_and)

plt.title("PA XOR")
print('[PA] Training XOR function')
train(x_input, y_xor)
