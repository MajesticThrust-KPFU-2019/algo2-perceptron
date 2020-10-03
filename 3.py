"""
Двухслойный и трехслойный перцептрон.

Задать веса произвольным образом. Посчитать вывод двухслойного и трехслойного
перцептрона. Посчитать для каждого ошибку (MSE).

У промежуточного слоя (веса 2*3) немножко перемешан вывод.

w00 w01 | x00 = x10
w10 w11 | x01 = x11

x10 x11 | v00 v01 v02 = x20
        | v10 v11 v12 = x21
                      = x22

w1 w2 w3 | x20 = y^
         | x21
         | x22

E = y - y^

Вариант 3.

Результат выписать в csv (разделитель = `,`). Дробные значения писать с `.`.
V23.csv, V33.csv для двухслойного и трехслойного перцептрона сответственно.

V23:
w00, w01, w10, w11, w1, w2, E

V33:
w00, w01, w10, w11, v00, v01, v02, v10, v11, v12, w1, w2, w3, E

y - ожидаемое, y^ - полученное.
delta_i = y - y^
E - sum(delta_i**2)
"""

from typing import Iterable
import numpy as np


class Perceptron:
    def __init__(self, layer_sizes: Iterable):
        self.layer_sizes = list(layer_sizes)

        rng = np.random.default_rng()
        self.weights = []
        for i in range(len(self.layer_sizes) - 1):
            l_prev = self.layer_sizes[i]
            l_next = self.layer_sizes[i + 1]
            weights = rng.random((l_next, l_prev))
            self.weights.append(weights)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def evaluate(self, x: np.array):
        """
        Рассчитывает результат нейросети для входного вектора x.

        x должен быть вектором-столбцом.
        """

        for W in self.weights:
            x = self._sigmoid(W.dot(x))

        return x


def main():
    # 2-layer perceptron
    p2 = Perceptron([2, 2, 1])

    # 3-layer perceptron
    p3 = Perceptron([2, 2, 3, 1])

    x_file = open("xdata.csv")
    y_file = open("ydata03.csv")

    err_p2 = 0
    err_p3 = 0
    for x_str, y_str in zip(x_file, y_file):
        x = np.array(list(map(float,
                              x_str.strip().split(";")))).reshape((2, 1))
        y = np.array(float(y_str.strip()))

        y_p2 = p2.evaluate(x)
        y_p3 = p3.evaluate(x)

        err_p2 += sum((y - y_p2)**2)
        err_p3 += sum((y - y_p3)**2)

    x_file.close()
    y_file.close()

    # write v23.csv
    # w00, w01, w10, w11, w1, w2, E
    with open("v23.csv", "w") as f:
        results = [
            p2.weights[0][0, 0], p2.weights[0][0, 1], p2.weights[0][1, 0],
            p2.weights[0][1, 0], p2.weights[1][0, 0], p2.weights[1][0, 1],
            err_p2[0]
        ]
        f.write("\n".join(map(str, results)))

    # write v33.csv
    # w00, w01, w10, w11, v00, v01, v02, v10, v11, v12, w1, w2, w3, E
    with open("v33.csv", "w") as f:
        results = [
            p3.weights[0][0, 0], p3.weights[0][0, 1], p3.weights[0][1, 0],
            p3.weights[0][1, 1], p3.weights[1].T[0, 0], p3.weights[1].T[0, 1],
            p3.weights[1].T[0, 2], p3.weights[1].T[1, 0],
            p3.weights[1].T[1, 1], p3.weights[1].T[1, 2], p3.weights[2][0, 0],
            p3.weights[2][0, 1], p3.weights[2][0, 2], err_p3[0]
        ]
        f.write("\n".join(map(str, results)))


if __name__ == "__main__":
    main()