"""Module defining inputters.
Inputters implement the logic of transforming raw data to vectorized inputs,
e.g., from a line of text to a sequence of embeddings.
"""
from ogan.inputters.inputter import load_fields_from_vocab, get_fields, save_fields_to_vocab, \
    build_dataset, build_vocab, merge_vocabs, OrderedIterator

from ogan.inputters.dataset_base import DatasetBase, PAD_WORD, BOS_WORD, \
    EOS_WORD, UNK

from ogan.inputters.text_dataset import SingleTurnDataset, ShardedTextCorpusIterator



__all__ = ['PAD_WORD', 'BOS_WORD', 'EOS_WORD', 'UNK', 'DatasetBase',
           'load_fields_from_vocab', 'get_fields',
           'save_fields_to_vocab', 'build_dataset',
           'build_vocab', 'merge_vocabs', 'OrderedIterator',
           'SingleTurnDataset', 'ShardedTextCorpusIterator']