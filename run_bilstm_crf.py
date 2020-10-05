# -*- coding: utf-8 -*-
# @Time    : 2020/10/1 14:20
# @Author  : ZiUNO
# @Email   : ziunocao@126.com
# @File    : run_bilstm_crf.py
# @Software: PyCharm

import torch

from model.bilstm_crf import BiLSTM_CRF


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
print("score: ", model(X, Y)[0])
print("predict: ", model(X)[1])
