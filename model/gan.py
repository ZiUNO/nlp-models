# -*- coding: utf-8 -*-
# @Time    : 2020/10/9 20:11
# @Author  : ZiUNO
# @Email   : ziunocao@126.com
# @File    : gan.py
# @Software: PyCharm
from torch import nn

from model.autoencoder import AutoEncoder
from model.crf import CRF

from model.embed import Embed
from model.pump import Pump


class Generator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, seq_length):
        super(Generator, self).__init__()
        self.seq_length = seq_length
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, start_emb, noise_state):
        sentence_state = noise_state
        word_emb = start_emb
        for _ in range(self.seq_length):
            word_emb, sentence_state = self.lstm(word_emb, sentence_state)
            word_emb = self.fc(word_emb)
        return sentence_state


class Discriminator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, seq_length):
        super(Discriminator, self).__init__()
        self.seq_length = seq_length
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, embedding_dim)
        self.final_fc = nn.Linear(hidden_dim, 2)
        self.cls = nn.Softmax(2)

    def forward(self, start_emb, sentence_state):
        output = start_emb
        lstm_feats = None
        for _ in range(self.seq_length):
            lstm_feats, _ = self.lstm(output, sentence_state)
            output = self.fc(lstm_feats)
        dis_output = self.final_fc(lstm_feats)
        cls = self.cls(dis_output)
        return cls


class GAN:
    def __init__(self, embedding_dim, hidden_dim, seq_length, pretrained):
        super(GAN, self).__init__()
        self.emb = Embed.from_pretrained(pretrained['Embed'])
        self.pum = Pump.from_pretrained(pretrained['Pump'])
        self.gen = Generator(embedding_dim, hidden_dim, seq_length)
        self.dis = Discriminator(embedding_dim, hidden_dim, seq_length)
        self.ae = AutoEncoder.from_pretrained(pretrained['AutoEncoder'])
        self.crf = CRF.from_pretrained(pretrained['CRF'])

    @property
    def embedding(self):
        return self.emb

    @property
    def pumping(self):
        return self.pum

    @property
    def generator(self):
        return self.gen

    @property
    def discriminator(self):
        return self.dis

    def encode(self, enc_input):
        return self.ae.encode(enc_input)

    def decode(self, dec_input):
        return self.ae.decode(dec_input)
