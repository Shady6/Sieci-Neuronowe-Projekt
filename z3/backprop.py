import copy
import enum
import math
import numpy as np
import matplotlib.pyplot as plt

ws = [
    np.array([
        [None, .86, .82],
        [None, -.16, -.51],
        [None, .86, -.89],
    ]),
    np.array([
        [.04],
        [-.43],
        [.48],
    ])
]

xs = [
    np.array([1, 0, 0]),
    np.array([1, 0, 1]),
    np.array([1, 1, 0]),
    np.array([1, 1, 1])
]

ys = [
    0,
    1,
    1,
    0,
]


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


gradients = []

for x, y in zip(xs, ys):
    _as = [x]
    zs = [None]

    for l, w in enumerate(ws):
        z_l = []
        for i in range(w.shape[1]):  # Simplify
            w_l = w[:, i]

            z_il = 0
            if w_l[0] != None:
                z_il = np.dot(_as[-1], w_l)

            z_l.append(z_il)
        zs.append(z_l)
        _as.append([sigmoid(x) if x != None else 1 for x in z_l])

    da = 2*(_as[-1][0] - y)
    da_prevs = _as[:][1:]
    da_prevs[-1][0] = da

    dws = copy.deepcopy(ws.copy())

    for l_rev, w in enumerate(reversed(ws)):
        l = len(ws) - l_rev
        for i in range(w.shape[1]):  # Simplify
            w_l = w[:, i]
            for j, one_w in enumerate(w_l):
                if not one_w:
                    continue
                da_dz = [sigmoid_derivative(x) for x in zs[l]]
                dz_dw = _as[l-1][j]
                dw = da_prevs[l-1][i] * da_dz[i] * dz_dw
                dws[l - 1][j, i] = dw

                if l-2 >= 0:
                    dz_da_prev = one_w
                    da_prev = da_prevs[l-1][i] * dz_da_prev
                    da_prevs[l-2][j] = da_prev

    gradients.append(dws)

new_weights = [np.array(ws[0][:, 1:], float), ws[1]]
alfa = 5
gradients_without_none = [
    [np.array(x[0][:, 1:], float), x[1]] for x in gradients]
total_gradient = [np.zeros(x.shape) for x in gradients_without_none[0]]

for partial_gradient in gradients_without_none:
    for i, layer_gradient in enumerate(partial_gradient):
        total_gradient[i] += layer_gradient

total_gradient = [g / len(gradients) for g in total_gradient]

# tutaj jakas padaka narazie
# diffs = [[]
#          for _ in range(sum([x.shape[0] + x.shape[1] for x in new_weights]))]
# for i in range(100):
#     old_weights = copy.deepcopy(new_weights)
#     new_weights = [w - alfa * g for w, g in zip(new_weights, total_gradient)]
#     weights_diff = [new - old for new, old in zip(new_weights, old_weights)]

#     i = 0
#     for ws in weights_diff:
#         for w in ws:
#             for x in w:
#                 diffs[i].append(x)
#                 i += 1


# for diff in diffs:
#     plt.plot(np.linspace(0, len(diff), len(diff)), diff)

# plt.show()
