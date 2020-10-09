# -*- coding: utf-8 -*-
# @Time    : 2020/10/9 20:11
# @Author  : ZiUNO
# @Email   : ziunocao@126.com
# @File    : gan.py
# @Software: PyCharm
import torch
from torch import nn

embedding_dim = 300  # depend on the autoencoder embedding_dim
START_EMBEDDING = torch.randn((1, embedding_dim))
STOP_EMBEDDING = torch.randn((1, embedding_dim))


# TODO append a crf after decoder's lstm : maybe cannot calculate the grad !
class Generator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(Generator, self).__init__()

    def forward(self, start_emb):
        sentence = start_emb
        return sentence


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

    def forward(self):
        pass
