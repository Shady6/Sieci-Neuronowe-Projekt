from math import floor
import numpy as np
import matplotlib.pyplot as plt


def _translate(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    valueScaled = float(value - leftMin) / float(leftSpan)

    return rightMin + (valueScaled * rightSpan)


def activ(x):
    return 1 if x > 0 else 0


def neuron_process(x, w):
    return activ(x.dot(w))


def calc_y_plot(curr_weights, x_plot):
    if curr_weights[-1] == 0:
        return [curr_weights[0] for _ in range(len(x_plot))]
    else:
        return -curr_weights[0]/curr_weights[-1] + \
            sum([-wj * x_plot / curr_weights[-1]
                 for wj in curr_weights[1:-1]])


def plot(x_input, x_plot, ys_plot, weights, iterations):
    plt.scatter(x_input[:, 2:], x_input[:, 1:2])
    # Lini narysowanych jest iterations+1, poniewaz
    # rysujemy jeszcze dla wag poczatkowych, czego nie
    # uznaje za iteracje
    plt.xlabel(f'Iterations: {iterations}\n' +
               f'Weights: {weights}')
    plt.tight_layout()

    for (i, y_plot) in enumerate(ys_plot):
        plt.plot(x_plot, y_plot, alpha=_translate(i, 0, len(ys_plot),
                 0.1, 1), linewidth=3 if i == len(ys_plot) - 1 else 1)

        x_annotate = _translate(i, 0, len(ys_plot), 0, 1)
        y_annotate = y_plot[floor(x_annotate * len(x_plot))]
        plt.annotate(i, xy=(x_annotate, y_annotate))

    plt.show()


def plot3d(x, x_plot, w, iterations, title):
    fig = plt.figure()
    fig.suptitle(f'{title}\n' +
                 f'Iterations: {iterations}\n' +
                 f'Weights: {w}')

    ax = fig.add_subplot(projection='3d')
    ax.scatter(x[:, 2:3],
               x[:, 1:2], x[:, -1:])

    y_plot = x_plot
    x_plot, y_plot = np.meshgrid(x_plot, y_plot)
    z_plot = -w[0]/w[3] - x_plot*w[1] / \
        w[3] - y_plot*w[2]/w[3]
    ax.plot_surface(x_plot, y_plot, z_plot)

    plt.show()
