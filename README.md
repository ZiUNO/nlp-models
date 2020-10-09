# nlp-models
##### (For CUDA)
## Models
- [x] <a href="#autoencoder">AutoEncoder</a>
- [x] <a href="#bilstmcrf">BiLSTM+CRF</a>
- [ ] GAN
- [ ] VAE
## Methods
### Optuna
```python
import optuna
import torch

from torch import nn
from model.autoencoder import AutoEncoder

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
```
---
### AutoEncoder
```python
from torch import nn

from model.autoencoder import AutoEncoder
import torch

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
for _ in range(epochs):

    target = model(enc_input, dec_input)
    loss = loss_function(target, tgt_output)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
with torch.no_grad():
    prediction = model(enc_input, dec_input)
```
---
### BiLSTM+CRF
```python
import torch

from model.bilstm_crf import BiLSTM_CRF

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

# training
for _ in range(epochs):
    model.zero_grad()
    loss, _ = model(X, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# predicting
with torch.no_grad():
    prediction = model(X)[1]
```
> https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#advanced-making-dynamic-decisions-and-the-bi-lstm-crf
---