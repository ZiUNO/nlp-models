# -*- coding: utf-8 -*-
# @Time    : 2020/10/1 14:21
# @Author  : ZiUNO
# @Email   : ziunocao@126.com
# @File    : bilstm_crf.py
# @Software: PyCharm

import torch
import torch.nn as nn

from model import *


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
        init_alphas = torch.full((self.batch_size, self.tagset_size), -1. * MAX_INT).cuda()
        # init_alphas[0][index of START_TAG] = 0
        init_alphas[:, -2] = 0.

        forward_var = init_alphas
        for i in range(self.seq_length):
            feat = feats[:, i]
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_scores = feat[:, next_tag].view(self.batch_size, -1).expand(self.batch_size, self.tagset_size)
                trans_scores = self.transitions[next_tag].expand(self.batch_size, self.tagset_size)
                next_tag_var = forward_var + trans_scores + emit_scores
                alphas_t.append(torch.logsumexp(next_tag_var, dim=1).view(self.batch_size, 1))
            forward_var = torch.cat(alphas_t).view(self.batch_size, -1)

        terminal_var = forward_var + self.transitions[-1].expand(self.batch_size, self.tagset_size)
        alphas = torch.logsumexp(terminal_var, dim=1)
        return alphas.sum()

    def _score_sentence(self, feats, tags):
        score = torch.zeros(self.batch_size).cuda()
        tags = torch.cat([torch.ones((self.batch_size, 1), dtype=torch.long).cuda() * (self.tagset_size - 2), tags],
                         dim=1)
        for j in range(self.seq_length):
            score = score + self.transitions[tags[:, j + 1], tags[:, j]].view(self.batch_size) + torch.diagonal(feats[:, j][:, tags[:, j + 1]])
        score = score + self.transitions[-1, tags[:, -1]]
        return score.sum()

    def _viterbi_decode(self, feats):
        init_vvars = torch.full((self.batch_size, self.tagset_size), -1. * MAX_INT).cuda()
        init_vvars[:, -2] = 0.

        forward_var = init_vvars

        backpointers = []
        for i in range(self.seq_length):
            bptrs_t = []
            viterbivars_t = []
            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var.view(self.batch_size, self.tagset_size) + self.transitions[next_tag].expand(
                    self.batch_size, self.tagset_size)
                best_tag_id = torch.argmax(next_tag_var, dim=1)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(torch.diagonal(next_tag_var[:, best_tag_id]))
            forward_var = (torch.cat(viterbivars_t).view(self.tagset_size, self.batch_size).transpose(0, 1) + feats[:,
                                                                                                              i]).view(
                self.batch_size, self.tagset_size)
            backpointers.append(bptrs_t)
        terminal_var = forward_var + self.transitions[-1].expand(self.batch_size, self.tagset_size)
        best_tag_id = torch.argmax(terminal_var, dim=1)
        path_scores = torch.diagonal(terminal_var[:, best_tag_id])
        best_paths = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            bptrs_t = torch.cat(bptrs_t).view(self.tagset_size, self.batch_size)
            best_tag_id = torch.diagonal(bptrs_t[best_tag_id])
            best_paths.append(best_tag_id)
        best_paths = torch.cat(best_paths).view(self.seq_length + 1, self.batch_size).transpose(0, 1)
        start = best_paths[:, -1].tolist()
        best_paths = best_paths[:, :-1]
        assert start == [self.tagset_size - 2] * self.batch_size
        best_paths = best_paths.tolist()
        for i, path in enumerate(best_paths):
            path.reverse()
            best_paths[i] = path
        path_scores = path_scores.tolist()
        return torch.tensor(path_scores, dtype=torch.float).sum(), best_paths

    def forward(self, X, Y=None):
        """

        :param X: torch.LongTensor([[1, 4, 3, 2, 6], [1, 4, 3, 0, 0]])
        :param Y: torch.LongTensor([[2, 1, 5, 2, 3], [2, 1, 5, 0, 0]])
        :return:
        """
        # BiLSTM
        self.seq_length = X.shape[1]
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
