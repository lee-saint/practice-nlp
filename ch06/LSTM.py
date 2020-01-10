import numpy as np

from common.functions import sigmoid


class LSTM:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        A = np.matmul(x, Wx) + np.matmul(h_prev, Wh) + b

        # slice
        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H]

        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        tahn_c_next = np.tahn(c_next)

        ds = dc_next + (dh_next * o) * (1 - tahn_c_next**2)

        dc_prev = ds * f

        di = ds * g
        df = ds * c_prev
        do = dh_next * tahn_c_next
        dg = ds * i

        di *= i * (1 - i)
        df *= f * (1 - f)
        do *= o * (1 - o)
        dg *= (1 - g**2)

        dA = np.hstack((df, dg, di, do))

        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = dA.sum(axis=0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        return dx, dh_prev, dc_prev


class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None
        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape  # 배치 개수, 시계열, 벡터 크기
        H = Wh.shape[0]     # 은닉층 벡터 크기

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')  # (배치 개수, 시계열, 은닉층 벡터 크기)로 은닉층 시계열 벡터 초기화

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')  # 전 시계열에서 상속받는 은닉층/기억 셸이 없으면 0으로 초기화

        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)  # 해당 계층을 LSTM으로 처리
            hs[:, t, :] = self.h                                         # 은닉층 시계열 벡터에 저장

            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape  # (배치 개수, 시계열, 은닉층 벡터 크기)
        D = Wx.shape[0]      # 단어 벡터 크기

        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0

        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)  # 파라미터로 들어온 dh에 한 시각 미래에서 넘어온 dh를 합산
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad  # 기울기 dWx, dWh, db를 계속 더함

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad  # 클래스 필드(기울기) 업데이트
        self.dh = dh  # dh 변수 저장
        return dxs

    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None


if __name__ == '__main__':
    pass
