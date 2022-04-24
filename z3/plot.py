import matplotlib.pyplot as plt
import numpy as np


def set_test_case_breakpoints(breakpoints):
    _, ax = plt.subplots()
    ax.set_xticks(breakpoints)
    ax.set_xticklabels([f'x{i}]' for i in range(len(breakpoints))])
    for b in breakpoints:
        plt.axvline(b, linestyle='--',)


def plot_weight_changes(w1s, w2s, breakpoints=[], title='', gradientPlot=False):
    if len(breakpoints):
        set_test_case_breakpoints(breakpoints)
    else:
        plt.subplots()

    y_plots = [[] for _ in range(9)]
    for w1_plot in w1s:
        i = 0
        for wvec in w1_plot:
            y_plots[0+i].append(wvec[0])
            y_plots[1+i].append(wvec[1])
            y_plots[2+i].append(wvec[2])
            i += 3

    for wvec in w2s:
        y_plots[6].append(wvec[0])
        y_plots[7].append(wvec[0])
        y_plots[8].append(wvec[0])

    y_labels = ['w1_01', 'w1_11', 'w1_21',
                'w1_02', 'w1_12', 'w1_22',
                'w2_00', 'w2_10', 'w2_20']
    x_plot = np.linspace(0, len(w1s), len(w1s))
    for y_plot, y_label in zip(y_plots, y_labels):
        if gradientPlot:
            y_label = 'd' + y_label
        plt.plot(x_plot, y_plot, label=y_label)

    plt.title(title)
    plt.legend(loc="upper left")


def plot_cost_changes(costs, breakpoints=[], title=''):
    if len(breakpoints):
        set_test_case_breakpoints(breakpoints)
    else:
        plt.subplots()

    x_plot = np.linspace(0, len(costs), len(costs))
    plt.plot(x_plot, costs)
    plt.title(title)
