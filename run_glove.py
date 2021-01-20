# -*- coding: utf-8 -*-
# @Time    : 2021/1/20 15:24
# @Author  : ZiUNO
# @Email   : ziunocao@126.com
# @File    : run_glove.py
# @Software: PyCharm
from random import randint

import torch
from tqdm import tqdm

from model import GloVe

vocab_size = 5
embedding_dim = 10
window_size = 3

max_sentence_length = 10
min_sentence_length = 5
sentences_size = 3
raw_data = [[randint(1, vocab_size) for _ in range(randint(min_sentence_length, max_sentence_length - 1))]
            for _ in range(sentences_size)]

glove = GloVe(vocab_size, embedding_dim, window_size)
matrix = glove.co_occurrence_matrix(raw_data)
optimizer = torch.optim.Adagrad(glove.parameters(), lr=0.05)

for i in tqdm(range(50)):
    optimizer.zero_grad()
    J = glove()
    J.backward()
    optimizer.step()
    if i % 5:
        print("J: ", J.data)
