import numpy as np

from common.util import preprocess, create_co_matrix


def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x**2)) + eps)
    ny = y / (np.sqrt(np.sum(y**2)) + eps)
    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    # 검색어 추출
    if query not in word_to_id:
        print(f'{query}를 찾을 수 없습니다.')
        return

    print('\n[query]', query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 코사인 유사도 계산
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # 코사인 유사도를 기준으로 내림차순으로 출력
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(f'{id_to_word[i]}: {similarity[i]}')

        count += 1
        if count >= top:
            return


def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100) == 0:
                    print('%.1f%% 완료' % (100*cnt / total))
    return M


if __name__ == '__main__':
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size)

    c0 = C[word_to_id['you']]  # 'you'의 단어 벡터
    c1 = C[word_to_id['i']]    # 'i'의 단어 벡터
    print(cos_similarity(c0, c1))

    most_similar('you', word_to_id, id_to_word, C, top=5)

    W = ppmi(C)
    
    np.set_printoptions(precision=3)  # 유효 자릿수를 세 자리로 표시
    print('동시발생 행렬')
    print(C)
    print('-'*50)
    print('PPMI')
    print(W)
