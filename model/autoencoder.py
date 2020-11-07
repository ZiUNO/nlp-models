# -*- coding: utf-8 -*-
# @Time    : 2020/10/8 19:52
# @Author  : ZiUNO
# @Email   : ziunocao@126.com
# @File    : autoencoder.py
# @Software: PyCharm
import torch
from torch import nn

from model import Model


class AutoEncoder(nn.Module, Model):
    def __init__(self, embedding_dim, hidden_dim):
        super(AutoEncoder, self).__init__()
        self.enc_cell = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.dec_cell = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, embedding_dim)
        self.enc_seq_length = self.dec_seq_length = 0

    def forward(self, enc_input, dec_input):
        """

        :param enc_input: enc_input.shape = [batch_size, seq_length, embedding_dim]
        :param dec_input: dec_input.shape = [batch_size, seq_length + 1, embedding_dim]
                          # dec_input[batch_index, 0] = <START>.embedding
        :return: tgt_output.shape = [batch_size, seq_length + 1, embedding_dim]
                 # tgt_output[batch_index, -1] = <STOP>.embedding
        """
        self.enc_seq_length = max(enc_input.shape[1], self.enc_seq_length)
        self.dec_seq_length = max(dec_input.shape[1], self.dec_seq_length)
        _, hidden_state = self.enc_cell(enc_input)
        dec_output, _ = self.dec_cell(dec_input, hidden_state)
        tgt_output = self.fc(dec_output)
        return tgt_output

    def encode(self, enc_input):
        _, sentence_vec = self.enc_cell(enc_input)
        return sentence_vec

    def decode(self, sentence_vec, start_emb):
        output = []
        out = start_emb
        for _ in range(self.dec_seq_length):
            dec_out, sentence_vec = self.dec_cell(out, sentence_vec)
            out = self.fc(dec_out)
            output.append(out)
        return torch.tensor(output, dtype=torch.float)
