import numpy as np

w1 = 1/3 * np.array(
    [
        [0, -2, 2],
        [-2, 0, -2],
        [2, -2, 0],
    ]
)
xs1 = [
    np.array([-1, -1, -1]),
    np.array([-1, -1, 1]),
    np.array([-1, 1, -1]),
    np.array([-1, 1, 1]),
    np.array([1, -1, -1]),
    np.array([1, -1, 1]),
    np.array([1, 1, -1]),
    np.array([1, 1, 1]),
]
b1 = np.array([0, 0, 0])

w2 = np.array(
    [
        [0, 1],
        [-1, 0],
    ]
)
xs2 = [
    np.array([-1, -1]),
    np.array([-1, 1]),
    np.array([1, -1]),
    np.array([1, 1]),
]
b2 = np.array([0, 0])

xs = [xs1, xs2]
ws = [w1, w2]
bs = [b1, b2]
