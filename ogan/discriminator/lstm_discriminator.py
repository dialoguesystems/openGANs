"""
Implementation of "Convolutional Sequence to Sequence Learning"
"""
import torch.nn as nn

SCALE_WEIGHT = 0.5 ** 0.5

class LSTM_Discriminator(nn.Module):
    """
    Encoder built on CNN
    """
    def __init__(self, embeddings, hidden_size, dropout,  target_size=1):
        super(LSTM_Discriminator, self).__init__()
        vocab_size = embeddings.word_lut.num_embeddings
        embedding_size = embeddings.word_lut.embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, target_size)
        self.softmax = nn.Softmax()

    def forward(self, input, length=None):
        embeddings = self.embeddings(input)
        hidden_states, last_state = self.lstm(embeddings)
        latent_vector = self.dropout(last_state[0])
        true = self.softmax(self.linear(latent_vector))
        return  true