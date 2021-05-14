# AllenNLP Model
## Embedder
- [x] GloVe Embedding (using average word embedding as char embedding)

### GloVe Embedding
#### run_glove_embedder.json
```json
{
  "token_embedders": {
    "tokens": {
      "type": "embedding",
      "trainable": false,
      "embedding_dim": 300,
      "pretrained_file": "path/glove.6B.300d.txt"
    }
  }
}
```
#### command line
```shell script
allennlp train -s allennlp_model/model -f --include-package allennlp_model.embedder.glove_embedder allennlp_model/run_glove_embedder.json
```