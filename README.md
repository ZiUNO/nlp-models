# nlp-models
##### (for CUDA)
## Dir (for test)
- allennlp_model: Model for AllenNLP
- pytorch_module
- data
    - glove
        - glove.6B.300d.txt
        - glove.840B.300d.txt
    - squad
        - train-v2.0.json
        - train-v2.0-test.json: Abridged version of train-v2.0.json
        - dev-v2.0.json
        - dev-v2.0-test.json: Abridged version of dev-v2.0.json
    - text8
---
### Module: pytorch_module
- [x] BiLSTM+CRF
- [x] Seq2Seq
### Model for AllenNLP: allennlp_model
- [x] GloVe Embedding (using average word embedding as char embedding)
## Reference
> https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#advanced-making-dynamic-decisions-and-the-bi-lstm-crf