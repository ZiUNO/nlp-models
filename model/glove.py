# -*- coding: utf-8 -*-
# @Time    : 2021/1/20 15:23
# @Author  : ZiUNO
# @Email   : ziunocao@126.com
# @File    : glove.py
# @Software: PyCharm

from model import *


class GloVe(Module):
    def __init__(self, vocab_size, embedding_dim, window_size):
        super(GloVe, self).__init__()
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.w = Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        self.w_ = Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        self.b = Embedding(vocab_size + 1, 1, padding_idx=0)
        self.b_ = Embedding(vocab_size + 1, 1, padding_idx=0)
        self.matrix = torch.zeros((self.vocab_size, self.vocab_size))
        self.x_max = 0

    def co_occurrence_matrix(self, raw_data: list):
        for sentence in raw_data:
            for pos in range(0, len(sentence)):
                first = sentence[pos]
                for i, second in enumerate(sentence[pos + 1: pos + self.window_size]):
                    self.matrix[first - 1][second - 1] += 1 / (1 + i)
                    if first != second:
                        self.matrix[second - 1][first - 1] += 1 / (1 + i)
        self.x_max = self.matrix.max()
        return self.matrix

    def x(self, i, j):
        return self.matrix[i - 1][j - 1]

    def weight_function_x(self, i, j):
        return self.x(i, j) / self.x_max

    def distance(self, i, j):
        return torch.dot(self.w(i), self.w_(j))

    def function_x(self, i, j):
        return torch.log(self.x(i, j))

    def forward(self):
        J = 0
        for i in range(1, self.vocab_size + 1):
            i = tensor(i, dtype=torch.long)
            for j in range(1, self.vocab_size + 1):
                j = tensor(j, dtype=torch.long)
                if self.x(i, j).tolist() == 0:
                    continue
                J += self.weight_function_x(i, j) * torch.pow(
                    self.distance(i, j) + self.b(i)[0] + self.b_(j)[0] - self.function_x(i, j), 2)
        return J
