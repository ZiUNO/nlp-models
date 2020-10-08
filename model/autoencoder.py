# -*- coding: utf-8 -*-
# @Time    : 2020/10/8 19:52
# @Author  : ZiUNO
# @Email   : ziunocao@126.com
# @File    : autoencoder.py
# @Software: PyCharm

from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(AutoEncoder, self).__init__()
        self.enc_cell = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.dec_cell = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, enc_input, dec_input):
        _, hidden_state = self.enc_cell(enc_input)
        dec_output, _ = self.dec_cell(dec_input, hidden_state)
        tgt_output = self.fc(dec_output)
        return tgt_output
