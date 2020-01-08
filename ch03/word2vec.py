import numpy as np
from common.layers import MatMul

c = np.array([[1, 0, 0, 0, 0, 0, 0]])
W = np.random.randn(7, 3)
print(W)

layer = MatMul(W)
h = layer.forward(c)
print(h)
