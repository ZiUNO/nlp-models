# -*- coding: utf-8 -*-
# @Time    : 2020/10/8 20:14
# @Author  : ZiUNO
# @Email   : ziunocao@126.com
# @File    : run_seq2seq.py
# @Software: PyCharm
from torch import nn

from pytorch_module.model import Seq2Seq
from tqdm import tqdm

import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

vocab_size = 20
embedding_dim = 32
hidden_dim = 20
batch_size = 10
seq_length = 20
beam_size = 2
# 1 = [START], 2 = [STOP]
enc_input = torch.randint(3, vocab_size, (batch_size, seq_length)).cuda()
dec_input = torch.cat((torch.ones(batch_size, 1).cuda() * 1,
                       enc_input), dim=-1).cuda()
tgt_output = torch.cat((enc_input,
                        torch.ones(batch_size, 1).cuda() * 2), dim=-1).cuda()

enc_input = enc_input.detach().type(torch.long)
dec_input = dec_input.detach().type(torch.long)
tgt_output = tgt_output.detach().type(torch.long)

model = Seq2Seq(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, beam_size=beam_size)
model = model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
loss_function = nn.CrossEntropyLoss().cuda()

epochs = 1000
for epoch in tqdm(range(epochs), desc='Training'):
    target = model(enc_input, dec_input)
    loss = 0
    for index in range(len(target)):
        loss += loss_function(target[index], tgt_output[index])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if not epoch % 20000:
        logging.info("loss: %s" % str(loss.data))

with torch.no_grad():
    prediction = model(enc_input, dec_input)
# logging.info("Prediction: %s" % str(prediction.tolist()))
# logging.info("GroundTruth: %s" % str(tgt_output.tolist()))
logging.info("Loss: %s" % str(loss_function(prediction[0], tgt_output[0]).data))
