import numpy as np
from common.functions import softmax
from ch06.RNNLM import Rnnlm
from ch06.better_rnnlm import BetterRnnlm
from dataset import ptb


class RnnlmGen(Rnnlm):
    def generate(self, start_id, skip_ids=None, sample_size=100):
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            score = self.predict(x)
            p = softmax(score.flatten())

            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids


class BetterRnnlmGen(BetterRnnlm):
    def generate(self, start_id, skip_ids=None, sample_size=100):
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            score = self.predict(x).flatten()
            p = softmax(score).flatten()

            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids


if __name__ == '__main__':
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    vocab_size = len(word_to_id)
    corpus_size = len(corpus)

    model = RnnlmGen()
    model.load_params('../ch06/Rnnlm.pkl')

    # 시작(start) 문자와 건너뜀(skip) 문자 설정
    start_word = 'you'
    start_id = word_to_id[start_word]
    skip_words = ['N', '<unk>', '$']
    skip_ids = [word_to_id[w] for w in skip_words]

    # 문장 생성
    word_ids = model.generate(start_id, skip_ids)
    txt = ' '.join([id_to_word[i] for i in word_ids])
    txt = txt.replace(' <eos>', '.\n')
    print(txt)

    better_model = BetterRnnlmGen()
    better_model.load_params('../ch06/BetterRnnlm.pkl')

    # 시작(start) 문자와 건너뜀(skip) 문자 설정
    start_word = 'you'
    start_id = word_to_id[start_word]
    skip_words = ['N', '<unk>', '$']
    skip_ids = [word_to_id[w] for w in skip_words]

    # 문장 생성
    word_ids = better_model.generate(start_id, skip_ids)
    txt = ' '.join([id_to_word[i] for i in word_ids])
    txt = txt.replace(' <eos>', '.\n')
    print(txt)

    better_model.reset_state()

    model.reset_state()

    start_words = 'the meaning of life is'
    start_ids = [word_to_id[w] for w in start_words.split(' ')]

    for x in start_ids[:-1]:
        x = np.array(x).reshape(1, 1)
        model.predict(x)

    word_ids = model.generate(start_ids[-1], skip_ids)
    word_ids = start_ids[:-1] + word_ids
    txt = ' '.join([id_to_word[i] for i in word_ids])
    txt = txt.replace(' <eos>', '.\n')
    print('-' * 50)
    print(txt)
