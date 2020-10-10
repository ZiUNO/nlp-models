# -*- coding: utf-8 -*-
# @Time    : 2020/10/10 19:58
# @Author  : ZiUNO
# @Email   : ziunocao@126.com
# @File    : pump.py
# @Software: PyCharm
from torch import nn

from model import Model


class Pump(nn.Module, Model):
    def __init__(self, embedding_dim, vocab_size):
        super(Pump, self).__init__()
        self.pump = nn.Linear(embedding_dim, vocab_size)

    def forward(self, pump_input):
        return self.pump(pump_input)