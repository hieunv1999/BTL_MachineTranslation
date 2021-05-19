import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.data import Field, BucketIterator
import numpy as np
import random
from tqdm import tqdm
import os
import re
import pandas as pd
from pyvi import ViTokenizer,ViPosTagger
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Pair:
  def __init__(self,sr,trg):
    self.src = sr
    self.trg = trg

def ckeck_replace()
def normalize_sent(sentence):
    sent = sentence.split(' ')
    tag_pos = ViPosTagger.postagging(sentence)
    for i in range(len(tag_pos[1])):
      if tag_pos[1][i]=='Np':
        sent[i] = 'Np_PPP'
    return ' '.join(sent)
def preprocess_sentence(w):
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = normalize_sent(w)
    w = w.lower().strip()
    w = remove_stopword(w)
    w = ViTokenizer.tokenize(w)
    return w
def remove_stopword(w):
    stop_word = '@#$%^&**()[]/<-_>\;:{}"'
    for i in stop_word:
        w = w.replace(i,'')
    return w
def get_data_raw(path):
    vi = []
    en = []
    list_folder = os.listdir(path)
    for folder in list_folder:
        if 'basic' in folder :
            list_file = os.listdir(os.path.join(path, folder))
            for file in list_file:
                link = os.path.join(path, folder, file)
                if '.vi' in file:
                    f = open(link, 'r', encoding='utf8')
                    for x in f:
                        vi.append(preprocess_sentence(x))
                if '.en' in file:
                    f = open(link, 'r', encoding='utf8')
                    for x in f:
                        en.append(preprocess_sentence(x))
    return vi, en
def get_data():
    vi = []
    en = []
    f = open('dataset/data.vi', 'r', encoding='utf8')
    for x in f:
        vi.append(preprocess_sentence(x))
    f = open('dataset/data.en', 'r', encoding='utf8')
    for x in f:
        en.append(preprocess_sentence(x))
    return vi,en
def get_datasets(lenght_sen):
    train = []
    # vie, eng = get_data()
    vie, eng = get_data_raw('dataset/MT-EV-VLSP2020')
    df = pd.DataFrame({'source':eng,'target':vie})
    source = Field(
    tokenize='basic_english', lower=True,init_token='<sos>', eos_token='<eos>'
    )
    target = Field(
    tokenize='basic_english', lower=True,init_token='<sos>', eos_token='<eos>'
    )
    english = df['source'].apply(lambda x: source.preprocess(x))
    vietnam = df['target'].apply(lambda x: target.preprocess(x))
    source.build_vocab(
        english
    )
    target.build_vocab(
        vietnam
    )
    source_vocab = source.vocab
    target_vocab = target.vocab
    for i in range(len(english)):
      e = list(map(lambda x: source_vocab.stoi[x],['<sos>']+english[i]+['<eos>']))
      v = list(map(lambda x: target_vocab.stoi[x], ['<sos>']+vietnam[i]+['<eos>']))
      if len(e)<=lenght_sen:
        e = np.array(e + (lenght_sen-len(e))*[1]).reshape((lenght_sen))
      else:
        e = np.array(e[:lenght_sen-1]+[3]).reshape((lenght_sen))
      if len(v)<=lenght_sen:
        v = np.array(v + (lenght_sen-len(v))*[1]).reshape((lenght_sen))
      else:
        v = np.array(v[:lenght_sen-1]+[3]).reshape((lenght_sen))
      train.append(Pair(e,v))
    return train,source,target

if __name__ == '__main__':
    get_data_raw('dataset/MT-EV-VLSP2020')