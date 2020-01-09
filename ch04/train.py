import numpy as np
import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from ch04.cbow import CBOW
from common.util import create_contexts_target
from dataset import ptb

if __name__ == '__main__':
    # 하이퍼파라미터 설정
    window_size = 5
    hidden_size = 100
    batch_size = 100
    max_epoch = 10

    # 데이터 읽기
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    vocab_size = len(word_to_id)

    contexts, target = create_contexts_target(corpus, window_size)

    # 모델 등 생성
    model = CBOW(vocab_size, hidden_size, window_size, corpus)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    # 학습 시작
    trainer.fit(x=contexts, t=target, max_epoch=max_epoch, batch_size=batch_size)
    trainer.plot()

    # 나중에 사용할 수 있도록 필요한 데이터 저장
    word_vecs = model.word_vecs
    params = {'word_vecs': word_vecs.astype(np.float16), 'word_to_id': word_to_id, 'id_to_word': id_to_word}
    pkl_file = 'cbow_params.pkl'
    with open(pkl_file, 'wb') as f:
        pickle.dump(params, f, -1)
