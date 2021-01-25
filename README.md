# nlp-models
##### (For CUDA)
## Models
- [x] <a href="#autoencoder">Seq2Seq</a>
- [x] <a href="#bilstmcrf">BiLSTM+CRF</a>
- [ ] <a href="#glove">GloVe</a>
## Methods
### Seq2Seq
```python
from torch import nn

from model import Seq2Seq
import torch


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
for epoch in range(epochs):
    target = model(enc_input, dec_input)
    loss = 0
    for index in range(len(target)):
        loss += loss_function(target[index], tgt_output[index])
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

from model import BiLSTM_CRF


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
### GloVe
```python
import torch

from model import GloVe

vocab_size = 5
embedding_dim = 2
window_size = 1

raw_data = [[5, 3, 3, 5, 4, 5], [4, 4, 1, 2, 2, 2, 2, 4, 4], [4, 3, 2, 5, 2]]

glove = GloVe(vocab_size, embedding_dim, window_size)
matrix = glove.co_occurrence_matrix(raw_data)
optimizer = torch.optim.Adagrad(glove.parameters(), lr=0.05)

for i in range(500):
    optimizer.zero_grad()
    J = glove()
    J.backward()
    optimizer.step()
    
with torch.no_grad():
    J = glove().data
```
