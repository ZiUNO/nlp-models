# -*- coding: utf-8 -*-
# @Time    : 2020/10/10 19:24
# @Author  : ZiUNO
# @Email   : ziunocao@126.com
# @File    : run_gan.py
# @Software: PyCharm

import torch

from model.gan import GAN

# TODO need to be changed
embedding_dim = 10  # depend on the autoencoder embedding_dim
hidden_dim = 16

batch_size = 4
seq_length = 20
vocab_size = 50
vocab = list(range(vocab_size))
START_INDEX = vocab_size - 2
STOP_INDEX = vocab_size - 1

# real_data without start or stop
real_data = torch.randint(vocab_size - 2, (batch_size, seq_length))
real_labels = torch.tensor([0, 1]).expand(batch_size, 2)

gan = GAN(embedding_dim, hidden_dim, seq_length,
          pretrained={'Embed': 'emb.pth',
                      'Pump': 'pum.pth',
                      'AutoEncoder': 'ae.pth',
                      'CRF': 'crf.pth'})

# Optimizers
optimizer_gen = torch.optim.Adam(gan.generator.parameters(), lr=0.01, weight_decay=1e-4)
optimizer_dis = torch.optim.Adam(gan.discriminator.parameters(), lr=0.01, weight_decay=1e-4)

START_EMBED = gan.embedding([START_INDEX])[0]
STOP_EMBED = gan.embedding([STOP_INDEX])[0]

dis_loss_func = torch.nn.BCELoss()

epochs = 100
for _ in range(epochs):
    # -----
    # Train generator
    # -----
    # fake_sentence_vecs
    fake_noise_state = (torch.randn((batch_size, 1, seq_length, hidden_dim)),
                        torch.randn((batch_size, 1, seq_length, hidden_dim)))
    fake_labels = torch.tensor([1, 0]).expand(batch_size, 2)

    optimizer_gen.zero_grad()
    # generator generates fake data to fool discriminator
    fake_gen_sentence_vecs = gan.generator(START_EMBED, fake_noise_state)
    fake_emb = gan.decode(fake_gen_sentence_vecs)
    enc_sentence_vecs = gan.encode(fake_emb)
    fake_dis = gan.discriminator(enc_sentence_vecs)

    fake_dis_loss = dis_loss_func(fake_dis, real_labels)

    fake_dis_loss.backward()
    optimizer_gen.step()

    # -----
    # Train discriminator
    # -----
    optimizer_dis.zero_grad()
    # discriminator classifies the real data to real labels
    real_emb = gan.embedding(real_data)
    real_sentence_vecs = gan.encode(real_emb)
    real_dis = gan.discriminator(real_sentence_vecs)
    real_dis_loss = dis_loss_func(real_dis, real_labels)
    # discriminator classifier the fake data to fake labels
    fake_dis_loss = dis_loss_func(fake_dis, fake_labels)

    dis_loss = (real_dis_loss + fake_dis_loss) / 2

    dis_loss.backward()
    optimizer_dis.step()
