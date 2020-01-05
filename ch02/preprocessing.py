# import re
import numpy as np


def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word


if __name__ == '__main__':

    text = 'You say goodbye and I say hello.'
    text = text.lower()
    text = text.replace('.', ' .')
    print(text)

    words = text.split(' ')
    # words = re.split('(\w+)', text)
    print(words)

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    print(id_to_word)
    print(word_to_id)

    print(id_to_word[1])
    print(word_to_id['hello'])

    corpus = [word_to_id[w] for w in words]
    corpus = np.array(corpus)
    print(corpus)

    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    print(corpus)
    print(word_to_id)
    print(id_to_word)
