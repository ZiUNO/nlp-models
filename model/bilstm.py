# -*- coding: utf-8 -*-
# @Time    : 2020/10/6 20:42
# @Author  : ZiUNO
# @Email   : ziunocao@126.com
# @File    : bilstm.py
# @Software: PyCharm

from model import *


class BiLSTM(Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(BiLSTM, self).__init__()

        tagset_size += 2

        self.hidden_dim = hidden_dim

        self.word_emb = Embedding(vocab_size, embedding_dim)
        self.bilstm = LSTM(embedding_dim, hidden_dim // 2,
                           num_layers=1, bidirectional=True)
        self.fc = Linear(hidden_dim // 2 * 2, tagset_size)

    def forward(self, X):
        """

        :param X: X.shape = [batch_size, seq_length]: batch_size * word_index_sequence
        :return: lstm_feats.shape = [batch_size, seq_length, tagset_size]: batch_size * seq_length * word2tag_feature_distribution
        """
        self.seq_length = X.shape[1]
        self.batch_size = X.shape[0]
        hidden = (torch.randn(2, self.seq_length, self.hidden_dim // 2).cuda(),
                  torch.randn(2, self.seq_length, self.hidden_dim // 2).cuda())
        emb = self.word_emb(X)
        lstm_out, (_, _) = self.bilstm(emb, hidden)
        lstm_feats = self.fc(lstm_out)
        return lstm_feats
