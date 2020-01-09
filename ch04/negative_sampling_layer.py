from collections import Counter

import numpy as np

from ch01.forward_net import cross_entropy_error
from common.layers import Embedding


class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # sigmoid의 출력
        self.t = None  # 정답 데이터

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))

        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)  # t=0이면 1-y, t=1이면 y

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        return dx


class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)

        self.cache = h, target_W
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh


class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        """
        초기화 함수
        :param corpus: 단어 ID 목록
        :param power: 확률분포에 제곱할 값
        :param sample_size: 추출할 샘플 사이즈
        """
        self.sample_size = sample_size
        self.vocab_size = None  # 총 어휘 사이즈
        self.word_p = None      # 확률분포 array

        counts = Counter()  # 단어 출현 빈도를 저장할 카운터 객체
        for word_id in corpus:  # 코퍼스에 단어가 등장할 때마다 해당 단어 카운터를 1 올린다!
            counts[word_id] += 1

        vocab_size = len(counts)  # 카운터의 키 개수가 총 어휘 사이즈
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)  # 확률분포표 초기화 (어휘 사이즈만큼)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]  # 카운터에 등록된 단어 등장 횟수를 확률분포표로 가져옴

        self.word_p = np.power(self.word_p, power)  # 입력받은 값을 제곱함
        self.word_p /= np.sum(self.word_p)  # 전체값으로 나눠야 확률이 되겠죠?

    def get_negative_sample(self, target):
        """
        정답(target)이 아닌 샘플을 self.sample_size 만큼 추출
        :param target: 정답인 값의 인덱스(의 배치 크기만큼의 배열)
        :return: 부정적 예 샘플(의 배치 크기만큼의 배열) (sample_size, batch_size)
        """
        batch_size = target.shape[0]  # 타겟 사이즈가 배치 크기

        negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)
        # 네거티브 샘플 초기화

        for i in range(batch_size):  # 각 타겟마다 반복
            p = self.word_p.copy()  # 확률
            target_idx = target[i]  # 정답 인덱스
            p[target_idx] = 0  # 정답은 추출하면 안되니까 확률을 0으로
            p /= p.sum()  # 다시 확률의 합을 1로 맞춤
            negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)

        return negative_sample


class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]
        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        # 긍정적 예 순전파
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)

        # 부정적 예 순전파
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[i + 1].forward(h, negative_target)
            loss += self.loss_layers[i + 1].forward(score, negative_label)

        return loss

    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)

        return dh


if __name__ == '__main__':
    # 0부터 9까지의 숫자 중 하나를 무작위로 샘플링
    print(np.random.choice(10))

    # words에서 하나만 무작위로 샘플링
    words = ['you', 'say', 'goodbye', 'I', 'hello', '.']
    print(np.random.choice(words))

    # 5개만 무작위로 샘플링(중복 있음)
    print(np.random.choice(words, size=5))

    # 5개만 무작위로 샘플링(중복 없음)
    print(np.random.choice(words, size=5, replace=False))

    # 확률분포에 따라 샘플링
    p = [0.5, 0.1, 0.05, 0.2, 0.05, 0.1]
    print(np.random.choice(words, p=p))

    p = [0.7, 0.29, 0.01]
    new_p = np.power(p, 0.75)
    new_p /= np.sum(new_p)
    print(new_p)

    corpus = np.array([0, 1, 2, 3, 4, 1, 2, 3])
    power = 0.75
    sample_size = 2

    sampler = UnigramSampler(corpus, power, sample_size)
    target = np.array([1, 3, 0])
    negative_sample = sampler.get_negative_sample(target)
    print(negative_sample)
