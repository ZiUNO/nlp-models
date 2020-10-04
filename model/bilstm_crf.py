# -*- coding: utf-8 -*-
# @Time    : 2020/10/1 14:21
# @Author  : ZiUNO
# @Email   : ziunocao@126.com
# @File    : bilstm_crf.py
# @Software: PyCharm

import torch
import torch.nn as nn
from multiprocessing.dummy import Pool as ThreadPool
from model import *


def _argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()


def _log_sum_exp(vec):
    max_score = vec[0, _argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()

        tagset_size += 2

        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.word_emb = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                              num_layers=1, bidirectional=True)
        self.fc = nn.Linear(hidden_dim // 2 * 2, tagset_size)
        # transitions[-2] : START_TAG
        # transitions[-1] : STOP_TAG
        self.transitions = nn.Parameter(torch.randn(tagset_size, tagset_size))

        self.transitions.data[-2, :] = - MAX_INT
        self.transitions.data[:, -1] = - MAX_INT

    def _forward_alg(self, feats):
        init_alphas = torch.full((self.batch_size, self.tagset_size), -1.0 * MAX_INT).cuda()
        # init_alphas[0][index of START_TAG] = 0
        init_alphas[:, -2] = 0

        forward_var = init_alphas

        def f(i):
            for feat in feats[i]:
                alphas_t = []
                for next_tag in range(self.tagset_size):
                    emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                    trans_score = self.transitions[next_tag].view(1, -1)
                    next_tag_var = forward_var[i:i + 1] + trans_score + emit_score
                    alphas_t.append(_log_sum_exp(next_tag_var).view(1))
                forward_var[i] = torch.cat(alphas_t).view(1, -1)[0]

        pool = ThreadPool(self.batch_size)
        pool.map(f, range(self.batch_size))
        pool.close()
        pool.join()

        terminal_var = torch.full((self.batch_size, self.tagset_size), -1.0 * MAX_INT).cuda()
        for i in range(self.batch_size):
            terminal_var[i] = forward_var[i] + self.transitions[-1]
        alphas = torch.zeros(self.batch_size, dtype=torch.float)
        for i in range(self.batch_size):
            alphas[i] = _log_sum_exp(terminal_var[i:i + 1])
        return alphas.mean()

    def _score_sentence(self, feats, tags):
        score = torch.zeros(self.batch_size)
        tags = torch.cat([torch.ones((self.batch_size, 1), dtype=torch.long) * (self.tagset_size - 2), tags], dim=1)

        def f(i):
            for j, feat in enumerate(feats[i]):
                score[i] = score[i] + self.transitions[tags[i][j + 1], tags[i][j]] + feat[tags[i][j + 1]]
            score[i] = score[i] + self.transitions[-1, tags[i][-1]]

        pool = ThreadPool(self.batch_size)
        pool.map(f, range(self.batch_size))
        pool.close()
        pool.join()
        return score.mean()

    def _viterbi_decode(self, feats):
        path_scores = []
        best_paths = []

        init_vvars = torch.full((self.batch_size, self.tagset_size), -1.0 * MAX_INT).cuda()
        terminal_var = torch.full((self.batch_size, self.tagset_size), -1.0 * MAX_INT).cuda()
        init_vvars[:, -2] = 0

        forward_var = init_vvars
        for i in range(self.batch_size):
            backpointers = []
            for feat in feats[i]:
                bptrs_t = []
                viterbivars_t = []

                for next_tag in range(self.tagset_size):
                    next_tag_var = forward_var[i:i + 1] + self.transitions[next_tag]
                    best_tag_id = _argmax(next_tag_var)
                    bptrs_t.append(best_tag_id)
                    viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
                forward_var[i] = (torch.cat(viterbivars_t) + feat).view(1, -1)
                backpointers.append(bptrs_t)
            terminal_var[i] = forward_var[i] + self.transitions[-1]
            best_tag_id = _argmax(terminal_var[i:i + 1])

            path_score = terminal_var[i][best_tag_id]

            best_path = [best_tag_id]
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)
            start = best_path.pop()
            assert start == self.tagset_size - 2
            best_path.reverse()
            path_scores.append(path_score)
            best_paths.append(best_path)
        return torch.tensor(path_scores, dtype=torch.float).mean(), best_paths

    def forward(self, X, Y=None):
        """

        :param X: torch.LongTensor([[1, 4, 3, 2, 6], [1, 4, 3, 0, 0]])
        :param Y: torch.LongTensor([[2, 1, 5, 2, 3], [2, 1, 5, 0, 0]])
        :return:
        """
        # BiLSTM
        self.seq_length = X.shape[-1]
        self.batch_size = X.shape[0]
        hidden = (torch.randn(2, self.seq_length, self.hidden_dim // 2).cuda(),
                  torch.randn(2, self.seq_length, self.hidden_dim // 2).cuda())
        emb = self.word_emb(X)
        lstm_out, (_, _) = self.bilstm(emb, hidden)
        lstm_feats = self.fc(lstm_out)
        # lstm_feats : the input of CRF
        tag_seq = Y
        if Y is not None:
            # training
            assert self.batch_size == Y.shape[0]
            forward_score = self._forward_alg(lstm_feats)
            gold_score = self._score_sentence(lstm_feats, tag_seq)
            score = forward_score - gold_score
        else:
            # predicting
            score, tag_seq = self._viterbi_decode(lstm_feats)
            tag_seq = torch.tensor(tag_seq, dtype=torch.long)
        return score, tag_seq
