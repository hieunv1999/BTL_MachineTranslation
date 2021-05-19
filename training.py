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

def create_model(source, target):
    embedding_dim = 256
    hidden_dim = 1024
    dropout = 0.5
    encoder = Encoder(len(source.vocab), embedding_dim, hidden_dim, n_layers=2, dropout_prob=dropout)
    one_step_decoder = OneStepDecoder(len(target.vocab), embedding_dim, hidden_dim, n_layers=2, dropout_prob=dropout)
    decoder = Decoder(one_step_decoder, device)
    model = EncoderDecoder(encoder, decoder)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    TARGET_PAD_IDX = target.vocab.stoi[target.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=TARGET_PAD_IDX)
    return model, optimizer, criterion

def train(train_iterator, valid_iterator, source, target, epochs):
    model, optimizer, criterion = create_model(source, target)

    clip = 1
    loss_min = 1000
    for epoch in range(1, epochs + 1):
        training_loss = []
        model.train()
        for i, (src,trg) in enumerate(train_iterator):
            src = src.to(device)
            trg = trg.to(device)
            optimizer.zero_grad()
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            training_loss.append(loss.item())
            if i%50 == 0:
                print('Epoch {} : Batch {}/{} Loss : {}'.format(epoch,i,len(train_iterator),loss.item()))
        n = len(training_loss)
        loss_epoch = np.sum([training_loss])/n
        print('----Loss epoch {} : {}'.format(epoch,loss_epoch))
        if epoch % 50 == 0 and loss_epoch < loss_min:
            loss_min = loss_epoch
            checkpoint = {
            'model_state_dict': model.state_dict(),
            'source': source.vocab,
            'target': target.vocab
            }

            torch.save(checkpoint, 'nmt-model-lstm-{}.pth'.format(epoch))

    return model
if __name__ == '__main__':
    train_iterator, source, target = get_datasets(16)
    engls = [feature.src for feature in train_iterator]
    viets = [feature.trg for feature in train_iterator]
    data_train = TensorDataset(torch.LongTensor(engls),(torch.LongTensor(viets)))
    data_train = DataLoader(data_train,batch_size = 128,drop_last=True)
    model = train(data_train, data_train, source, target, epochs=100)
    # checkpoint = {
    #     'model_state_dict': model.state_dict(),
    #     'source': source.vocab,
    #     'target': target.vocab
    # }

    # torch.save(checkpoint, 'nmt-model-lstm-{}.pth'.format(epoch))
