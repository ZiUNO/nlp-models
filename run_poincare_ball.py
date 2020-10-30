# -*- coding: utf-8 -*-
# @Time    : 2020/10/30 14:20
# @Author  : ZiUNO
# @Email   : ziunocao@126.com
# @File    : run_poincare_ball.py
# @Software: PyCharm
import torch
import logging
from geoopt import PoincareBall
from geoopt.optim import RiemannianAdam
from tqdm import tqdm
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

batch_size = 2
seq_length = 3
embedding_dim = 3
vocab = ['<pad>', 'i', 'love', 'apple', 'banana', 'hate']
vocab_size = len(vocab)

X_p = torch.tensor([[1, 2, 3], [1, 2, 4]])  # positive samples: i love apple, i love banana
X_p_ = torch.tensor([[1, 2, 4], [1, 2, 3]])
X_n = torch.tensor([[1, 5, 4], [1, 5, 3]])  # negative samples: i hate banana, i hate apple
X_n_ = torch.tensor([[1, 5, 3], [1, 5, 4]])
emb = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

model = PoincareBall()
optimizer = RiemannianAdam(emb.parameters(), lr=0.001)
epochs = 1000

with torch.no_grad():
    logging.info("X_p: %s" % str(X_p))
    logging.info("X_p: %s" % str(X_n))
    logging.info("Distance between X_p and X_n: %s" % str(model.dist(emb(X_p), emb(X_n)).data))

for epoch in tqdm(range(epochs)):
    emb_x_p_1 = emb(X_p)
    emb_x_n_1 = emb(X_n)
    emb_x_p_2 = emb(X_p_)
    emb_x_n_2 = emb(X_n_)
    optimizer.zero_grad()
    diff = model.dist(emb_x_p_1, emb_x_n_1)
    same_1 = model.dist(emb_x_n_1, emb_x_n_2)
    same_2 = model.dist(emb_x_p_1, emb_x_p_2)
    loss = - diff.sum() + same_1.sum() + same_2.sum()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    logging.info("X_p: %s" % str(X_p))
    logging.info("X_p: %s" % str(X_n))
    e_emb_x_p = emb(X_p)
    e_emb_x_n = emb(X_n)
    p_emb_x_p = model.projx(e_emb_x_p)
    p_emb_x_n = model.projx(e_emb_x_n)
    logging.info("Distance between X_p and X_n: %s" % str(model.dist(e_emb_x_p, e_emb_x_n).data))
    e_emb_w = emb.weight.data
    p_emb_w = model.projx(emb.weight).data
    logging.info("Weight of emb: %s" % str(e_emb_w))
    logging.info("Projx weight of emb: %s" % str(p_emb_w))
    logging.info("Euclidean embedding of X_p: %s" % str(e_emb_x_p))
    logging.info("Euclidean embedding of X_n: %s" % str(e_emb_x_n))
    logging.info("Poincare embedding of X_p: %s" % str(p_emb_x_p))
    logging.info("Poincare embedding of X_n: %s" % str(p_emb_x_n))


def show(number, data_1, data_2, c_1, c_2, labels_1, labels_2):
    plt.figure(number)
    pic = plt.subplot(1, 1, 1, projection='3d')
    for i in range(data_1.shape[0]):
        pic.scatter(data_1[i, :, 0], data_1[i, :, 1], data_1[i, :, 2], c=c_1)
        pic.scatter(data_2[i, :, 0], data_2[i, :, 1], data_2[i, :, 2], c=c_2)
        for j in range(data_1.shape[1]):
            pic.text(data_1[i, j, 0], data_1[i, j, 1], data_1[i, j, 2], labels_1[i][j], verticalalignment='center',
                     horizontalalignment='center', color=c_1)
            pic.text(data_2[i, j, 0], data_2[i, j, 1], data_2[i, j, 2], labels_2[i][j], verticalalignment='center',
                     horizontalalignment='center', color=c_2)


x_p_labels = [['i', 'love', 'apple'], ['i', 'love', 'banana']]
x_n_labels = [['i', 'hate', 'banana'], ['i', 'hate', 'apple']]
# plt embed
pic = plt.subplot(1, 1, 1, projection='3d')
pic.scatter(e_emb_w[:, 0], e_emb_w[:, 1], e_emb_w[:, 2], c='r')
pic.scatter(p_emb_w[:, 0], p_emb_w[:, 1], p_emb_w[:, 2], c='b')
for e in range(e_emb_w.shape[0]):
    pic.text(e_emb_w[e, 0], e_emb_w[e, 1], e_emb_w[e, 2], vocab[e])
    pic.text(p_emb_w[e, 0], p_emb_w[e, 1], p_emb_w[e, 2], vocab[e])

# plt in euclidean
show(2, e_emb_x_p, e_emb_x_n, 'y', 'g', x_p_labels, x_n_labels)
# plt in poincare
show(3, p_emb_x_p, p_emb_x_n, 'y', 'g', x_p_labels, x_n_labels)

plt.show()
