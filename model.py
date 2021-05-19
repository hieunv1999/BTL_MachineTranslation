from preprocess import get_datasets
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset,DataLoader
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Encoder(nn.Module):
    def __init__(self, vocab_len, embedding_dim, hidden_dim, n_layers, dropout_prob):
        super().__init__()
        self.embedding = nn.Embedding(vocab_len, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers ,dropout=dropout_prob, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
    def forward(self, input_batch):
        embed = self.dropout(self.embedding(input_batch))
        outputs, (hidden, cell) = self.rnn(embed)
        return hidden, cell


class OneStepDecoder(nn.Module):
    def __init__(self, input_output_dim, embedding_dim, hidden_dim, n_layers, dropout_prob, batch_first=True):
        super().__init__()
        self.input_output_dim = input_output_dim
        self.embedding = nn.Embedding(input_output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim,hidden_dim, n_layers, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, input_output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, target_token, hidden, cell):
        target_token = target_token.unsqueeze(0)
        embedding_layer = self.dropout(self.embedding(target_token))
        output, (hidden, cell) = self.rnn(embedding_layer, (hidden, cell))
        linear = self.fc(output.squeeze(0))

        return linear, hidden, cell


class Decoder(nn.Module):
    def __init__(self, one_step_decoder, device):
        super().__init__()
        self.one_step_decoder = one_step_decoder
        self.device = device

    def forward(self, target, hidden, cell, teacher_forcing_ratio=0.5):
        batch_size, target_len = target.shape[0], target.shape[1]
        target_vocab_size = self.one_step_decoder.input_output_dim
        predictions = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        input = target[:, 0]
        for t in range(1, target_len):
            predict, hidden, cell = self.one_step_decoder(input, hidden, cell)
            predictions[:,t,:] = predict
            input = predict.argmax(1)
            do_teacher_forcing = random.random() < teacher_forcing_ratio
            input = target[:,t] if do_teacher_forcing else input

        return predictions


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        hidden, cell = self.encoder(source)
        outputs = self.decoder(target, hidden, cell, teacher_forcing_ratio)

        return outputs