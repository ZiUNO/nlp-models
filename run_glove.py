# -*- coding: utf-8 -*-
# @Time    : 2021/1/20 15:24
# @Author  : ZiUNO
# @Email   : ziunocao@126.com
# @File    : run_glove.py
# @Software: PyCharm
import logging
from random import randint

import torch
from tqdm import tqdm

from model import GloVe

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

vocab_size = 5
embedding_dim = 2
window_size = 3

raw_data = [[5, 3, 3, 5, 4, 5], [4, 4, 1, 2, 2, 2, 2, 4, 4], [4, 3, 2, 5, 2]]

glove = GloVe(vocab_size, embedding_dim, window_size)
matrix = glove.co_occurrence_matrix(raw_data)
optimizer = torch.optim.Adagrad(glove.parameters(), lr=0.05)

for i in tqdm(range(500)):
    optimizer.zero_grad()
    J = glove()
    J.backward()
    optimizer.step()
    if not i % 100:
        logging.info("J: %s" % str(J.data))

with torch.no_grad():
    logging.info("J: %s" % str(glove().data))
