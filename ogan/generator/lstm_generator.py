import torch.nn as nn

class LSTM_Generator(nn.Module):
    def __init__(self, hidden_size, embeddings, memory):
        super(LSTM_Generator, self).__init__()
        vocab_size = embeddings.word_lut.num_embeddings
        embedding_size = embeddings.word_lut.embedding_dim

        self.memory_reader = self.make_memory_reader(memory)

        self.embeddings = embeddings
        self.decoder = self.maker_decoder(embedding_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax()

    def forward(self, input):
        latent_vector = self.memory_reader(input)
        embeddings = self.embeddings(input)
        hidden_states, last_state = self.decoder(latent_vector, embeddings)

        return hidden_states


    def sample(self, source):
        target = source
        return target

    def make_memory_reader(self, memory):
        return memory

    def maker_decoder(self, embedding_size, hidden_size):
        decoder = nn.LSTM(embedding_size, hidden_size)
        return decoder