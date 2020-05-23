#!/usr/bin/python3

import sys;
from os.path import exists, join;
from math import ceil;
import numpy as np;
import cv2;
import tensorflow as tf;
from models import CRNN;
from tokenizer import Tokenizer;

class TextRecognizer(object):

  def __init__(self):

    self.tokenizer = Tokenizer();
    # NOTE: extra class for blank index
    self.crnn = CRNN(self.tokenizer.size() + 1);
    if exists(join('model', 'crnn.h5')):
      self.crnn = tf.keras.models.load_model(join('model','crnn.h5'), compile = False);

  def resize(self, img):
    
    scale = img.shape[0] / 32;
    height = 32;
    width = int(img.shape[1] / scale);
    img = cv2.resize(img, (width, height));
    new_width = 4 * ceil(width / 4);
    if new_width > width:
      img = np.concatenate([img, np.zeros(32, new_width - width, 3)], axis = 1);
    return img;

  def recognize(self, img, preprocess = True):

    if preprocess == True:
      input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
      input = self.resize(input);
      inputs = (tf.cast(tf.expand_dims(input, axis = 0), dtype = tf.float32) / 255. - 0.5) * 2; # input.shape = (1, seq_length, 32);
    else:
      inputs = img;
    pred = self.crnn(inputs); # pred.shape = (batch, seq_length, num_class)
    decoded = tf.keras.backend.ctc_decode(pred, [pred.shape[1]], greedy = True);
    tokens = decoded[0][0]; # the most probabable path.shape = (seq_length, 1)
    prob = decoded[1][0]; # the probability of the most probable path.shape = (1,)
    tokens = tf.squeeze(tokens, axis = 1); # tokens.shape = (seq_length)
    return self.tokenizer.translate(tokens), prob;

if __name__ == "__main__":

  if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " <image>");
    exit();
  img = cv2.imread(sys.argv[1]);
  if img is None:
    print('failed to open image!');
    exit();
  text_recognizer = TextRecognizer();
  text, _ = text_recognizer.recognize(img);
  print(text);
