import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    np.random.seed(666)
    x = np.random.randn(10, 2)      # 입력(mini-batch)
    W1 = np.random.randn(2, 4)      # 가중치
    b1 = np.random.randn(4)         # 편향
    W2 = np.random.randn(4, 3)
    b2 =  np.random.randn(3)

    h = np.matmul(x, W1) + b1
    a = sigmoid(h)
    s = np.matmul(a, W2) + b2

    print(W1)
    print(b1)
    print(x)
    print(h)
    print(a)
    print(s)
