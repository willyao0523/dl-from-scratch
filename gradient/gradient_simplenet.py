import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class SimpleNet:
    def __init__(self) -> None:
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = SimpleNet()


def f(w): return net.loss(x, t)


dW = numerical_gradient(f, net.W)

print(dW)
