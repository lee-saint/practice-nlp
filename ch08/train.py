import matplotlib.pyplot as plt
import numpy as np

from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from ch08.attention_seq2seq import AttentionSeq2seq
from ch07.seq2seq import Seq2seq
from ch07.peeky_seq2seq import PeekySeq2seq

if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = sequence.load_data('date.txt')
    char_to_id, id_to_char = sequence.get_vocab()

    # 입력 문장 반전
    x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

    # 하이퍼파라미터 설정
    vocab_size = len(char_to_id)
    wordvec_size = 16
    hidden_size = 256
    batch_size = 128
    max_epoch = 10
    max_grad = 5.0

    model_a = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)
    # model_b = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)
    # model_c = Seq2seq(vocab_size, wordvec_size, hidden_size)
    optimizer_a = Adam()
    optimizer_b = Adam()
    optimizer_c = Adam()

    trainer_a = Trainer(model_a, optimizer_a)
    # trainer_b = Trainer(model_b, optimizer_b)
    # trainer_c = Trainer(model_c, optimizer_c)

    a_acc_list = []
    # b_acc_list = []
    # c_acc_list = []
    for epoch in range(max_epoch):
        trainer_a.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad)
        # trainer_b.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad)
        # trainer_c.fit(x_train, t_train, max_epoch=1, batch_size=batch_size, max_grad=max_grad)
        correct_num_a = 0
        # correct_num_b = 0
        # correct_num_c = 0
        for i in range(len(x_test)):
            question, correct = x_test[[i]], t_test[[i]]
            verbose = i < 10
            correct_num_a += eval_seq2seq(model_a, question, correct, id_to_char, verbose, is_reverse=True)
            # correct_num_b += eval_seq2seq(model_b, question, correct, id_to_char, False, is_reverse=True)
            # correct_num_c += eval_seq2seq(model_c, question, correct, id_to_char, False, is_reverse=True)

        acc_a = float(correct_num_a) / len(x_test)
        # acc_b = float(correct_num_b) / len(x_test)
        # acc_c = float(correct_num_c) / len(x_test)
        a_acc_list.append(acc_a)
        # b_acc_list.append(acc_b)
        # c_acc_list.append(acc_c)
        print('val acc %.3f%%' % (acc_a * 100))

    model_a.save_params()

    x = np.arange(max_epoch)
    plt.plot(x, a_acc_list, label='attention')
    # plt.plot(x, b_acc_list, label='peeky')
    # plt.plot(x, c_acc_list, label='baseline')
    plt.legend()
    plt.show()
