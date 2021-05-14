# -*- coding: utf-8 -*-
# @Time    : 2021/5/14 3:40 下午
# @Author  : ZiUNO
# @Email   : ziunocao@126.com
# @File    : glove_vocab.py
# @Software: PyCharm
import logging
from typing import Optional

import numpy
import torch
from allennlp.common import Tqdm
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import TokenEmbedder, TimeDistributed
from allennlp.modules.token_embedders.embedding import EmbeddingsTextFile
from allennlp.nn import util
from overrides import overrides
from torch.nn.functional import embedding

logger = logging.getLogger(__name__)


@TokenEmbedder.register("glove_embedding")
class GloVeEmbedding(TokenEmbedder):
    def __init__(
            self,
            embedding_dim: int,
            num_embeddings: int = None,
            projection_dim: int = None,
            weight: torch.FloatTensor = None,
            padding_index: int = None,
            trainable: bool = True,
            max_norm: float = None,
            norm_type: float = 2.0,
            scale_grad_by_freq: bool = False,
            sparse: bool = False,
            vocab_namespace: str = "tokens",
            pretrained_file: str = None,
            vocab: Vocabulary = None,
    ) -> None:
        super().__init__()

        if num_embeddings is None and vocab is None:
            raise ConfigurationError(
                "Embedding must be constructed with either num_embeddings or a vocabulary."
            )

        _vocab_namespace: Optional[str] = vocab_namespace
        if num_embeddings is None:
            num_embeddings = vocab.get_vocab_size(_vocab_namespace)  # type: ignore
        else:
            # If num_embeddings is present, set default namespace to None so that extend_vocab
            # call doesn't misinterpret that some namespace was originally used.
            _vocab_namespace = None  # type: ignore

        self.num_embeddings = num_embeddings
        self.padding_index = padding_index
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._vocab_namespace = _vocab_namespace
        self._pretrained_file = pretrained_file

        self.output_dim = projection_dim or embedding_dim

        if weight is not None and pretrained_file:
            raise ConfigurationError(
                "Embedding was constructed with both a weight and a pretrained file."
            )

        elif pretrained_file is not None:

            if vocab is None:
                raise ConfigurationError(
                    "To construct an Embedding from a pretrained file, you must also pass a vocabulary."
                )

            # If we're loading a saved model, we don't want to actually read a pre-trained
            # embedding file - the embeddings will just be in our saved weights, and we might not
            # have the original embedding file anymore, anyway.

            # TODO: having to pass tokens here is SUPER gross, but otherwise this breaks the
            # extend_vocab method, which relies on the value of vocab_namespace being None
            # to infer at what stage the embedding has been constructed. Phew.
            weight = _read_embeddings_from_text_file(
                pretrained_file, embedding_dim, vocab, vocab_namespace
            )
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)

        elif weight is not None:
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)

        else:
            weight = torch.FloatTensor(num_embeddings, embedding_dim)
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)
            torch.nn.init.xavier_uniform_(self.weight)

        # Whatever way we have constructed the embedding, it should be consistent with
        # num_embeddings and embedding_dim.
        if self.weight.size() != (num_embeddings, embedding_dim):
            raise ConfigurationError(
                "A weight matrix was passed with contradictory embedding shapes."
            )

        if self.padding_index is not None:
            self.weight.data[self.padding_index].fill_(0)

        if projection_dim:
            self._projection = torch.nn.Linear(embedding_dim, projection_dim)
        else:
            self._projection = None

    @overrides
    def get_output_dim(self) -> int:
        return self.output_dim

    @overrides
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens may have extra dimensions (batch_size, d1, ..., dn, sequence_length),
        # but embedding expects (batch_size, sequence_length), so pass tokens to
        # util.combine_initial_dims (which is a no-op if there are no extra dimensions).
        # Remember the original size.
        original_size = tokens.size()
        tokens = util.combine_initial_dims(tokens)

        embedded = embedding(
            tokens,
            self.weight,
            padding_idx=self.padding_index,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

        # Now (if necessary) add back in the extra dimensions.
        embedded = util.uncombine_initial_dims(embedded, original_size)

        if self._projection:
            projection = self._projection
            for _ in range(embedded.dim() - 2):
                projection = TimeDistributed(projection)
            embedded = projection(embedded)
        return embedded


def _read_embeddings_from_text_file(
        file_uri: str, embedding_dim: int, vocab: Vocabulary, namespace: str = "tokens"
) -> torch.FloatTensor:
    """
    Read pre-trained word vectors from an eventually compressed text file, possibly contained
    inside an archive with multiple files. The text file is assumed to be utf-8 encoded with
    space-separated fields: [word] [dim 1] [dim 2] ...

    Lines that contain more numerical tokens than `embedding_dim` raise a warning and are skipped.

    The remainder of the docstring is identical to `_read_pretrained_embeddings_file`.
    """
    tokens_to_keep = set(vocab.get_index_to_token_vocabulary(namespace).values())
    vocab_size = vocab.get_vocab_size(namespace)
    char_embeddings = {}
    embeddings = {}

    # First we read the embeddings from the file, only keeping vectors for the words we need.
    logger.info("Reading pretrained embeddings from file")

    with EmbeddingsTextFile(file_uri) as embeddings_file:
        for line in Tqdm.tqdm(embeddings_file):
            token = line.split(" ", 1)[0]
            if token in tokens_to_keep:
                fields = line.rstrip().split(" ")
                if len(fields) - 1 != embedding_dim:
                    # Sometimes there are funny unicode parsing problems that lead to different
                    # fields lengths (e.g., a word with a unicode space character that splits
                    # into more than one column).  We skip those lines.  Note that if you have
                    # some kind of long header, this could result in all of your lines getting
                    # skipped.  It's hard to check for that here; you just have to look in the
                    # embedding_misses_file and at the model summary to make sure things look
                    # like they are supposed to.
                    logger.warning(
                        "Found line with wrong number of dimensions (expected: %d; actual: %d): %s",
                        embedding_dim,
                        len(fields) - 1,
                        line,
                    )
                    continue

                vector = numpy.asarray(fields[1:], dtype="float32")
                for char in list(token):
                    if char in char_embeddings:
                        char_embeddings[char] = (char_embeddings[char][0] + vector, char_embeddings[char][1] + 1)
                    else:
                        char_embeddings[char] = (vector, 1)
                embeddings[token] = vector

    if not embeddings:
        raise ConfigurationError(
            "No embeddings of correct dimension found; you probably "
            "misspecified your embedding_dim parameter, or didn't "
            "pre-populate your Vocabulary"
        )

    char_embeddings = {char: char_embeddings[char][0] / char_embeddings[char][1] for char in char_embeddings}
    chars = set(char_embeddings.keys())

    all_embeddings = numpy.asarray(list(embeddings.values()))
    embeddings_mean = float(numpy.mean(all_embeddings))
    embeddings_std = float(numpy.std(all_embeddings))
    # Now we initialize the weight matrix for an embedding layer, starting with random vectors,
    # then filling in the word vectors we just read.
    logger.info("Initializing pre-trained embedding layer")
    embedding_matrix = torch.FloatTensor(vocab_size, embedding_dim).normal_(
        embeddings_mean, embeddings_std
    )
    num_tokens_found = 0
    index_to_token = vocab.get_index_to_token_vocabulary(namespace)
    for i in range(vocab_size):
        token = index_to_token[i]

        # If we don't have a pre-trained vector for this word, we'll just leave this row alone,
        # so the word has a random initialization.
        if token in embeddings:
            embedding_matrix[i] = torch.FloatTensor(embeddings[token])
            num_tokens_found += 1
        elif len(set(token) - chars) == 0:
            embedding_matrix[i] = torch.FloatTensor([char_embeddings[char] for char in list(token)]).sum(dim=-2)
            num_tokens_found += 1
        else:
            logger.debug(
                "Token %s was not found in the embedding file. Initialising randomly.", token
            )

    logger.info(
        "Pretrained embeddings were found for %d out of %d tokens", num_tokens_found, vocab_size
    )

    return embedding_matrix
