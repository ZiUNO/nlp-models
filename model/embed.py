# -*- coding: utf-8 -*-
# @Time    : 2020/10/10 19:58
# @Author  : ZiUNO
# @Email   : ziunocao@126.com
# @File    : embed.py
# @Software: PyCharm
from torch import nn

from model import Model


class Embed(nn.Module, Model):
    def __init__(self, vocab_size, embedding_dim):
        super(Embed, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, embed_input):
        return self.embed(embed_input)