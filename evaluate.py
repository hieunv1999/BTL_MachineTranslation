import torch
import re
from torchtext.data.metrics import bleu_score
from pyvi import ViTokenizer, ViPosTagger
from model import Encoder, Decoder, OneStepDecoder, EncoderDecoder
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
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

def normal_text(sentence):
    x = sentence.replace('i\'m', 'i am')
    x = x.replace('won\'t', 'will not')
    x = x.replace('\'ll', ' will')
    x = x.replace('it\'s', 'it is')
    x = x.replace('don\'t', 'do not')
    x = x.replace('can\'t', 'can not')
    x = x.replace('n\'t', ' not')
    x = x.replace('\'re', ' are')
    x = x.replace('he\'s', 'he is')
    x = x.replace('she\'s', 'she is')
    x = x.replace('where\'s', 'where is')
    x = x.replace('there\'s', 'there is')
    x = x.replace('that\'s', 'that is')
    x = x.replace('let\'s', 'let us')
    x = x.replace('\'d', ' would')
    x = x.replace('\'ve', ' have')
    x = x.replace('who\'s', 'who is')
    x = x.replace('what\'s', 'what is')
    x = x.replace('"', '')
    x = x.replace(' \' ', ' ')
    return x

def preprocess_name(sent):
    sent = sent.split(' ')
    name_first = None
    name_second = None
    for i in range(len(sent)):
        if sent[i].istitle() and i < 2:
            name_first = sent[i]
            sent[i] = 'i'
        elif sent[i].istitle():
            name_second = sent[i]
            sent[i] = 'you'
    return ' '.join(sent), name_first, name_second


def preprocess_date(sent):
    pattern_1 = "\d{1,4}[/-_]\d{1,4}[/-_]\d{1,4}"
    pattern_2 = '((january|february|march|april|may|june|july|august|september|october|november|december)[ ,]\d{1,2}(th|st|rd|nd)[ ,]\d{4})'
    pattern_4 = '((january|february|march|april|may|june|july|august|september|october|november|december)[ ,]\d{1,4})'
    date_pattern = None
    dates_1 = re.findall(pattern_1, sent)
    dates_2 = re.findall(pattern_2, sent)
    dates_4 = re.findall(pattern_4, sent)
    if dates_1:
        date_pattern = dates_1[0]
        sent = (sent.replace(date_pattern, '_datetime_'))
    elif dates_2:
        date_pattern = dates_2[0][0]
        sent = (sent.replace(date_pattern, '_datetime_'))
    elif dates_4:
        date_pattern = dates_4[0][0]
        sent = (sent.replace(date_pattern, '_datetime_'))
    return sent, date_pattern


def post_process(predicted, name_fisrt, name_second, date):
    dict_month = {'january': 'tháng 1', 'february': 'tháng 2', 'march': 'tháng 3', 'april': 'tháng 4', 'may': 'tháng 5',
                  'june':
                      'tháng 6', 'july': 'tháng 7', 'august': 'tháng 8', 'september': 'tháng 9', 'october': 'tháng 10',
                  'november': 'tháng 11',
                  'december': 'tháng 12'}
    if '_datetime_' in predicted and date == None:
        index = predicted.index('_datetime_')
        predicted[index] = 'thời gian'
    for word in predicted:
        if (word == 'tôi' or word == 'anh') and name_fisrt is not None:
            index = predicted.index(word)
            predicted[index] = name_fisrt
        if (word == 'bạn' or word == 'em') and name_second is not None:
            index = predicted.index(word)
            predicted[index] = name_second
        if word == '_datetime_':
            index = predicted.index(word)
            predicted[index] = date
    output = ' '.join(predicted)
    for word in output.split(' '):
        if word in list(dict_month.keys()):
            output = output.replace(word, dict_month[word])
    output = output.replace('<eos>', '')
    output = output.replace('_', ' ')
    return output



def predict(model, source_vocab, target_vocab, sentence, debug=False):
    sentence, name_st ,name_sd = preprocess_name(sentence)
    print(sentence,name_st,name_sd)
    sentence, date = preprocess_date(sentence)
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = normal_text(sentence)
    tokens = ['<sos>'] + sentence.lower().split(' ') + ['<eos>']
    src_indexes = [source_vocab.stoi[to] for to in tokens]
    if len(src_indexes) < 16:
        src_indexes = src_indexes + (16 - len(src_indexes)) * [1]
    else:
        src_indexes = src_indexes[:15] + [3]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    src_tensor = torch.reshape(src_tensor, (1, 16))
    model.eval()
    hidden, cell = model.encoder(src_tensor)
    trg_index = [target_vocab.stoi['<sos>']]
    next_token = torch.LongTensor(trg_index).to(device)
    outputs = []
    trg_indexes = []
    with torch.no_grad():
        for _ in range(30):
            output, hidden, cell = model.decoder.one_step_decoder(next_token, hidden, cell)
            next_token = output.argmax(1)
            trg_indexes.append(next_token.item())
            predicted = target_vocab.itos[output.argmax(1).item()]
            if predicted == '<eos>':
                break
            else:
                outputs.append(predicted)
    predicted_words = [target_vocab.itos[i] for i in trg_indexes]
    predicted_words = post_process(predicted_words, name_st, name_sd , date)
    return predicted_words


def remove_stopword(w):
    stop_word = '@#$%^&**()[]/<->\;:{}"'
    for i in stop_word:
        w = w.replace(i, '')
    return w


def preprocessing(token):
    token = re.sub(r"([?.!,¿])", r" \1 ", token)
    token = re.sub(r'[" "]+', " ", token)
    token = remove_stopword(token)
    token = token.lower().strip()
    return token
def pre_candidate(token):
    token = re.sub(r"([?.!,¿])", r" \1 ", token)
    token = re.sub(r'[" "]+', " ", token)
    token = token.lower().strip()
    return [token.split(' ')]

if __name__ == '__main__':
    checkpoint_file = 'nmt-model-lstm-90.pth'
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    source_vocab = checkpoint['source']
    target_vocab = checkpoint['target']
    model = create_model_for_inference(source_vocab, target_vocab)
    model.load_state_dict(checkpoint['model_state_dict'])
    # in_test = list(open('dataset/data.en',encoding='utf8'))
    # out_test = list(open('dataset/data.vi',encoding='utf8'))
    # scores = []
    # for i in tqdm(range(len(in_test))):
    #     sent = preprocessing(in_test[i].replace('\n',''))
    #     candidate = predict(model, source_vocab, target_vocab, 'npppp kept feeling it.').strip().split(' ')
    #     reference = pre_candidate(out_test[i].replace('\n',''))
    #     score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    #     scores.append(score)
    # print(sum(scores)/len(scores))
    sentence = 'i\'m in the dining room.'
    print(predict(model,source_vocab,target_vocab,sentence))
