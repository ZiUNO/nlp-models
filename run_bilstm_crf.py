# -*- coding: utf-8 -*-
# @Time    : 2020/10/1 14:20
# @Author  : ZiUNO
# @Email   : ziunocao@126.com
# @File    : run_bilstm_crf.py
# @Software: PyCharm

import torch
from tqdm import tqdm

import logging
from model import BiLSTM_CRF

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')

# X[i][j] = 0 : padding
# Y[i][j] = 0 : padding

vocab_size = 7
tagset_size = 6
embedding_dim = 5
hidden_dim = 8
X = torch.LongTensor([[1, 4, 3, 2, 6], [1, 4, 3, 0, 0]]).cuda()
Y = torch.LongTensor([[2, 1, 5, 2, 3], [2, 1, 5, 0, 0]]).cuda()
model = BiLSTM_CRF(vocab_size=vocab_size, tagset_size=tagset_size,
                   embedding_dim=embedding_dim, hidden_dim=hidden_dim)
model = model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

epochs = 500

for epoch in tqdm(range(epochs), desc='Training'):
    model.zero_grad()
    loss, _ = model(X, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    logging.info('Predicting: %s' % str(model(X)[1].tolist()))
