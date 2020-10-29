# -*- coding: utf-8 -*-
# @Time    : 2020/10/28 10:38
# @Author  : ZiUNO
# @Email   : ziunocao@126.com
# @File    : run_poincare.py
# @Software: PyCharm
import torch
import logging

from tqdm import tqdm

from model.poincare import poincare_distance

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

batch_size = 2
seq_length = 4
embed_dim = 2
vocab_size = 7
X = torch.tensor([[2, 1, 4, 3], [3, 5, 2, 1]], dtype=torch.long)
Y = torch.tensor([[5, 2, 1, 1], [2, 1, 2, 6]], dtype=torch.long)
emb = torch.nn.Embedding(vocab_size, embed_dim)

optimizer = torch.optim.Adam(emb.parameters(), lr=0.001)
logging.info("X: %s" % str(X))
logging.info("Y: %s" % str(Y))

epochs = 10000

with torch.no_grad():
    emb_x = emb(X)
    emb_y = emb(Y)
    poincare_dis = poincare_distance(emb_x, emb_y)
    logging.info("Poincare distance before training: %s" % poincare_dis)


for epoch in tqdm(range(epochs), desc='Training'):
    emb_x = emb(X)
    emb_y = emb(Y)
    optimizer.zero_grad()
    poincare_dis = poincare_distance(emb_x, emb_y)

    loss = - torch.sum(poincare_dis)
    if not epoch % 1000:
        logging.info(loss.data)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    emb_x = emb(X)
    emb_y = emb(Y)
    poincare_dis = poincare_distance(emb_x, emb_y)
    logging.info("Poincare distance: %s" % poincare_dis)
