#!/usr/bin/python3 
import tensorflow as tf;
import pandas as pd;

class Tokenizer(object):

  def __init__(self, vocab_file = 'vocab.pkl'):

    self.vocab = pd.read_pickle('vocab.pkl');

  def tokenize(self, str):

    tokens = list();
    for ch in str:
      token = self.vocab[self.vocab['character'] == ch]['token'].iloc[0];
      tokens.append(token);
    return tf.constant(tokens, dtype = tf.int64);

  def translate(self, tokens):

    s = list();
    for token in tokens:
      try:
        ch = self.vocab[self.vocab['token'] == token]['character'].iloc[0];
      except:
        print(self.vocab['token'] == token);
        exit(1);
      s.append(ch);
    return ''.join(s);

  def size(self):

    return len(self.vocab);

if __name__ == "__main__":

  tokenizer = Tokenizer();
  print(tokenizer.translate([0,1,2,3]));
  print(tokenizer.tokenize('你好世界'));
  print(tokenizer.size());
