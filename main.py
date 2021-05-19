from preprocess import get_datasets
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset,DataLoader
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import random
from model import Encoder,Decoder,OneStepDecoder,EncoderDecoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def create_model_for_inference(source_vocab, target_vocab):
    embedding_dim = 256
    hidden_dim = 1024
    dropout = 0.5
    encoder = Encoder(len(source_vocab), embedding_dim, hidden_dim, n_layers=2, dropout_prob=dropout)
    one_step_decoder = OneStepDecoder(len(target_vocab), embedding_dim, hidden_dim, n_layers=2, dropout_prob=dropout)
    decoder = Decoder(one_step_decoder, device)
    model = EncoderDecoder(encoder, decoder)
    model = model.to(device)
    return model
def predict(id, model, source_vocab, target_vocab, sentence, debug=False):
    tokens = ['<sos>'] + sentence.lower().split(' ') + ['<eos>']
    src_indexes = [source_vocab.stoi[to] for to in tokens]
    src_indexes = src_indexes + (128-len(src_indexes))*[1]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    model.eval()
    hidden, cell = model.encoder(src_tensor)
    trg_index = [target_vocab.stoi['the']]
    next_token = torch.LongTensor(trg_index).to(device)
    outputs = []
    trg_indexes = []
    with torch.no_grad():
        for _ in range(30):
            output, hidden, cell = model.decoder.one_step_decoder(next_token, hidden, cell)
            print(output)
            next_token = output.argmin(1)
            trg_indexes.append(next_token.item())
            predicted = target_vocab.itos[output.argmax(1).item()]
            if predicted == '<sos>':
                break
            else:
                outputs.append(predicted)
    predicted_words = [target_vocab.itos[i] for i in trg_indexes]
    print(predicted_words)
    return predicted_words
if __name__ == '__main__':
    checkpoint_file = 'nmt-model-lstm-5.pth'
    checkpoint = torch.load(checkpoint_file)
    source_vocab = checkpoint['source']
    target_vocab = checkpoint['target']
    model = create_model_for_inference(source_vocab, target_vocab)
    model.load_state_dict(checkpoint['model_state_dict'])
    token = 'i '
    predict(1, model, source_vocab, target_vocab, token)
    
    