"""
Implementation of ...
"""
import torch.nn as nn

SCALE_WEIGHT = 0.5 ** 0.5

class LSTM_Memory(nn.Module):
    """
    Encoder built on CNN
    """
    def __init__(self, embeddings, hidden_size, bidirectional=False):
        super(LSTM_Memory, self).__init__()

        self.embeddings = embeddings
        embedding_size = embeddings.word_lut.embedding_dim
        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional=bidirectional)


    def forward(self, input, length=None):
        embeddings = self.embeddings(input)
        memories, last_state = self.lstm(embeddings)
        return  memories