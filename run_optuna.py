# -*- coding: utf-8 -*-
# @Time    : 2020/10/9 16:27
# @Author  : ZiUNO
# @Email   : ziunocao@126.com
# @File    : run_optuna.py
# @Software: PyCharm

import optuna

from torch import nn

from model.autoencoder import AutoEncoder

import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

batch_size = 200
seq_length = 20


def objective(trial):
    embedding_dim = trial.suggest_int('embedding_dim', 16, 32)
    hidden_dim = trial.suggest_int('hidden_dim', 16, 32)
    epochs = trial.suggest_int('epochs', 10, 1000)

    enc_input = torch.randn(batch_size, seq_length, embedding_dim).cuda()
    dec_input = torch.cat((torch.randn(1, 1, embedding_dim).expand(batch_size, 1, embedding_dim).cuda(),
                           enc_input), dim=1).cuda()
    tgt_output = torch.cat((enc_input,
                            torch.randn(1, 1, embedding_dim).expand(batch_size, 1, embedding_dim).cuda()), dim=1).cuda()

    model = AutoEncoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = nn.MSELoss().cuda()

    for _ in range(epochs):
        target = model(enc_input, dec_input)
        loss = loss_function(target, tgt_output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        prediction = model(enc_input, dec_input)
    loss_score = loss_function(prediction, tgt_output).data

    return loss_score


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

trial = study.best_trial
best_params = trial.params.items()

for key, value in best_params:
    logging.info("{}: {}".format(key, value))
