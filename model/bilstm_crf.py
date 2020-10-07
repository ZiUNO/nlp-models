# -*- coding: utf-8 -*-
# @Time    : 2020/10/1 14:21
# @Author  : ZiUNO
# @Email   : ziunocao@126.com
# @File    : bilstm_crf.py
# @Software: PyCharm

import torch.nn as nn

from model.bilstm import BiLSTM
from model.crf import CRF


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()

        tagset_size += 2
        self.bilstm = BiLSTM(embedding_dim=embedding_dim, tagset_size=tagset_size,
                             vocab_size=vocab_size, hidden_dim=hidden_dim)
        self.crf = CRF(tagset_size=tagset_size)

    def forward(self, X, Y=None):
        """

        :param X: X.shape = [batch_size, seq_length]: batch_size * word_index_sequence
        :param Y: Y.shape = [batch_size, seq_length]: batch_size * tag_sequence
        :return: score.shape = [1]: sequence_score,
                 tag_seq = [batch_size, seq_length]: batch_size * prediction_tag_sequence
        """
        lstm_feats = self.bilstm(X)  # the input of CRF
        score, tag_seq = self.crf(lstm_feats, Y)
        return score, tag_seq
