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


def activ(x):
    return 1 if x > 0 else -1


def will_stabilize(w):
    for i, row in enumerate(w):
        for j, wij in enumerate(row):
            if (i == j and wij < 0) or w[i, j] != w[j, i]:
                return False
    return True


for i in range(len(xs)):
    print(f'Wektory wejściowe o długości: {ws[i].shape[0]}\n' +
          f'Wagi:\n {ws[i]}\n')
    if not will_stabilize(ws[i]):
        print('Sieć się nie ustabilizuje')

    stabilized = []
    periodic = []

    for x in xs[i]:
        v = x
        vs = [v]

        while True:
            u = np.dot(ws[i], v) + bs[i]
            v_next = np.array([activ(ui) for ui in u])
            vs.append(v_next)

            if np.array_equal(v, v_next):
                stabilized.append((x, v))
                break

            if np.array_equal(v_next, vs[0]):
                periodic.append((x, vs))
                break

            v = v_next

    stable_points = {tuple(s[1]) for s in stabilized}
    print(f'Punkty stabilne sieci:\n {stable_points}\n')

    print('Dane wektory wejściowe stabilizują się w następujący sposób:')
    [print(f'{s[0]} -> {s[1]}') for s in stabilized]
    print()

    print('Dane wektory wejściowe posiadają okresową konfigurację:')
    [print(f'{s[0]} -> {[r.tolist() for r in s[1]]}') for s in periodic]
    print()
