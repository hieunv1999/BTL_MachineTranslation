from keras.layers import Embedding,LSTM,Dropout,Dense,Layer
from keras import Model,Input
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
import keras.backend as K
import collections
import numpy as np
import time
from nltk.translate.bleu_score import corpus_bleu
  def eval(self, dataset):
    # get the source words and target_word_labels for the eval dataset
    source_words, target_words_labels = dataset
    vocab = self.target_dict.vocab

    # using the same encoding network used during training time, encode the training
    encoder_outputs, state_h,state_c = self.encoder_model.predict(source_words,batch_size=self.batch_size)
    # for max_target_step steps, feed the step target words into the decoder.
    predictions = []
    step_target_words = np.ones([source_words.shape[0],1]) * self.SOS
    for _ in range(self.max_target_step):
      
      step_decoder_outputs, state_h,state_c = self.decoder_model.predict([step_target_words,state_h,state_c,encoder_outputs],batch_size=self.batch_size)
      step_target_words = np.argmax(step_decoder_outputs,axis=2)
      predictions.append(step_target_words)

    # predictions is a [time_step x batch_size x 1] array. We use get_target_sentence() 
    #to recover the batch_size sentences
    candidates = self.get_target_sentences(np.concatenate(predictions,axis=1),vocab)
    references = self.get_target_sentences(target_words_labels,vocab,reference=True)

    # score using nltk bleu scorer
    score = corpus_bleu(references,candidates)
    print("Model BLEU score: %.2f" % (score*100.0))
    print("===== PRINTING OUTPUT====")
    for i in range(5):
      print("{} Sample".format(i+1))
      print("Predicted Sentence: "+" ".join(candidates[i]))
      print("Actual Sentence: "+" ".join(references[i][0]))
    print("========================")
    # Here we are printing the predicted sentence and actual sentence for each each epochs


def main(source_path, target_path, use_attention):
  max_example = 30000
  #use_attention = True
  print('loading dictionaries')
  train_data, dev_data, test_data, source_dict, target_dict = load_dataset(source_path,target_path,max_num_examples=max_example)
  print("read %d/%d/%d train/dev/test batches" % (len(train_data[0]),len(dev_data[0]), len(test_data[0])))

  model = NmtModel(source_dict,target_dict,use_attention)
  model.build()
  model.train(train_data,dev_data,test_data,10)