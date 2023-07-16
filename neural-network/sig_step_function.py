import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def step_function(x):
    return np.array(x > 0, dtype=np.int)

X = np.arange(-5.0, 5.0, 0.1)
Y1 = sigmoid(X)
Y2 = step_function(X)

plt.plot(X, Y1)
plt.plot(X, Y2, 'k--')
plt.ylim(-0.1, 1.1)
plt.show()