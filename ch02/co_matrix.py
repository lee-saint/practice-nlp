import numpy as np
from common.util import preprocess


def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


if __name__ == '__main__':
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)

    print(corpus)
    print(id_to_word)

    C = np.array([
        [0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0]
    ], dtype=np.int32)

    print('C[0] =', C[0])
    print('C[4] =', C[4])
    print('C[word_to_id["goodbye"]] =', C[word_to_id['goodbye']])

    co_matrix = create_co_matrix(corpus, 7)
    print(co_matrix)
