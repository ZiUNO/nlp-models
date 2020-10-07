# nlp-models
##### (For CUDA)
- [x] BiLSTM+CRF
- [ ] Transformer
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
for epoch in range(epochs):
    model.zero_grad()
    loss, _ = model(X, Y)
    loss.backward()
    optimizer.step()

# predicting
with torch.no_grad():
    prediction = model(X)[1]
```
> https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#advanced-making-dynamic-decisions-and-the-bi-lstm-crf
---
### Transformer