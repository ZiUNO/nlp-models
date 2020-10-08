# -*- coding: utf-8 -*-
# @Time    : 2020/10/8 20:14
# @Author  : ZiUNO
# @Email   : ziunocao@126.com
# @File    : run_autoencoder.py
# @Software: PyCharm
from torch import nn

from model.autoencoder import AutoEncoder
from tqdm import tqdm

import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')

embedding_dim = 32
hidden_dim = 20
batch_size = 8
seq_length = 5
enc_input = torch.randn(batch_size, seq_length, embedding_dim).cuda()
dec_input = torch.cat((torch.randn(1, 1, embedding_dim).expand(batch_size, 1, embedding_dim).cuda(),
                       enc_input), dim=1).cuda()
tgt_output = torch.cat((enc_input,
                        torch.randn(1, 1, embedding_dim).expand(batch_size, 1, embedding_dim).cuda()), dim=1).cuda()

model = AutoEncoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
model = model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
loss_function = nn.MSELoss()

epochs = 100000
for epoch in tqdm(range(epochs), desc='Training'):

    target = model(enc_input, dec_input)
    loss = loss_function(target, tgt_output)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if not epoch % 20000:
        logging.info("loss: %s" % str(loss.data))

prediction = model(enc_input, dec_input)
# logging.info("Prediction: %s" % str(prediction.tolist()))
# logging.info("GroundTruth: %s" % str(tgt_output.tolist()))
logging.info("Loss: %s" % str(loss_function(prediction, tgt_output).data))
