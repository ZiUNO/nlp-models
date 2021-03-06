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
            for pos in range(len(sentence)):
                first = sentence[pos]
                for i, second in enumerate(sentence[max(0, pos - self.window_size): pos]):
                    self.matrix[first - 1][second - 1] += 1 / (pos - max(pos - self.window_size, 0) - i)
                    if first != second:
                        self.matrix[second - 1][first - 1] += 1 / (pos - max(pos - self.window_size, 0) - i)
                for i, second in enumerate(sentence[pos + 1: pos + self.window_size + 1]):
                    self.matrix[first - 1][second - 1] += 1 / (1 + i)
                    if first != second:
                        self.matrix[second - 1][first - 1] += 1 / (1 + i)
        self.x_max = self.matrix.max()
        return self.matrix

    def x(self, i, j):
        return self.matrix[i - 1][j - 1]

    def weight_function_x(self, i, j):
        """
        f(Xij)
        @param i:
        @param j:
        @return: f(Xij)
        """
        return torch.pow(self.x(i, j) / self.x_max, 0.75)

    def distance(self, i, j):
        """
        wi^T times w_j
        @param i:
        @param j:
        @return: the inner product of wi and w_j
        """
        return torch.dot(self.w(i), self.w_(j))

    def function_x(self, i, j):
        """
        log(Xij)
        @param i:
        @param j:
        @return: log(Xij)
        """
        return torch.log(self.x(i, j))

    def forward(self):
        """
        Calculate GloVe loss J
        @return: J
        """
        J = 0
        for i in range(self.vocab_size):
            i = tensor(i + 1, dtype=torch.long)
            for j in range(self.vocab_size):
                j = tensor(j + 1, dtype=torch.long)
                if self.x(i, j).tolist() == 0:
                    continue
                J += self.weight_function_x(i, j) * torch.pow(
                    self.distance(i, j) + self.b(i)[0] + self.b_(j)[0] - self.function_x(i, j), 2)
        return J
