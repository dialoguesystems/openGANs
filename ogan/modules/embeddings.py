""" Embeddings module """
import math

import torch
import torch.nn as nn

from ogan.modules.util_class import Elementwise


class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.
    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2) *
                             -(math.log(10000.0) / dim)).float())
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb):
        emb = emb * math.sqrt(self.dim)
        emb = emb + self.pe[:emb.size(0)]
        emb = self.dropout(emb)
        return emb


class Embeddings(nn.Module):
    """
    Words embeddings for encoder/decoder.
    Additionally includes ability to add sparse input features
    based on "Linguistic Input Features Improve Neural Machine Translation"
    :cite:`sennrich2016linguistic`.
    Args:
        word_vec_size (int): size of the dictionary of embeddings.
        word_padding_idx (int): padding index for words in the embeddings.
        position_encoding (bool): see :obj:`onmt.modules.PositionalEncoding`
        dropout (float): dropout probability.
    """

    def __init__(self, word_vocab_size,
                 word_vec_size,
                 word_padding_idx,
                 position_encoding=False,
                 dropout=0,
                 sparse=False):

        self.word_padding_idx = word_padding_idx

        # The embedding matrix look-up tables.
        # self.embedding_size = word_vec_size
        embeddings = nn.Embedding(word_vocab_size, word_vec_size, padding_idx=word_padding_idx, sparse=sparse)


        super(Embeddings, self).__init__()
        self.make_embedding = nn.Sequential()
        self.make_embedding.add_module('word_lut', embeddings)

        if position_encoding:
            pe = PositionalEncoding(dropout, word_vec_size)
            self.make_embedding.add_module('pe', pe)

    @property
    def word_lut(self):
        """ word look-up table """
        return self.make_embedding[0]

    @property
    def emb_luts(self):
        """ embedding look-up table """
        return self.make_embedding[0]

    def load_pretrained_vectors(self, emb_file, fixed):
        """Load in pretrained embeddings.
        Args:
          emb_file (str) : path to torch serialized embeddings
          fixed (bool) : if true, embeddings are not updated
        """
        if emb_file:
            pretrained = torch.load(emb_file)
            self.word_lut.weight.data.copy_(pretrained)
            if fixed:
                self.word_lut.weight.requires_grad = False

    def forward(self, source):
        """
        Computes the embeddings for words and features.
        Args:
            source (`LongTensor`): index tensor `[len x batch x 1]`
        Return:
            `FloatTensor`: word embeddings `[len x batch x embedding_size]`
        """
        emb = self.make_embedding(source)

        return emb