# nlp-models
##### (For CUDA)
- [x] <a href="#autoencoder">AutoEncoder</a>
- [x] <a href="#bilstmcrf">BiLSTM+CRF</a>
- [ ] GAN
- [ ] VAE
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