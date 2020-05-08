#!/usr/bin/python3

import sys;
from os.path import exists, join;
import cv2;
import tensorflow as tf;
from models import OCR;
from tokenizer import Tokenizer;

class TextRecognizer(object):

  def __init__(self):

    self.tokenizer = Tokenizer();
    self.ocr = OCR(self.tokenizer.size());
    if exists(join('model', 'ocr.h5')):
      self.ocr = tf.keras.models.load_model(join('model','ocr.h5'), compile = False);

  def resize(self, img):
    
    scale = img.shape[0] / 32;
    height = img.shape[0] / scale;
    width = img.shape[1] / scale;
    img = cv2.resize(img, (width, height));
    new_width = 8 * round(width / 8);
    if width >= new_width:
      img = cv2.resize(img, (new_width, height));
    else:
      padding = 255 * np.ones((height, new_width - width, 3), dtype = np.uint8);
      left_width = np.random.randint(low = 0, high = padding.shape[1]);
      left_padding = padding[:,:left_width,:];
      right_padding = padding[:,left_width:,:];
      img = np.concatenate([left_padding, img, right_padding], axis = 1);
    return img;

  def recognize(self, img, preprocess = True):

    if preprocess == True:
      input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
      input = self.resize(input);
      inputs = tf.cast(tf.expand_dims(input, axis = 0), dtype = tf.float32) / 255.; # input.shape = (1, seq_length, 32);
    else:
      inputs = img;
    logits = self.ocr(inputs);
    decoded, _ = tf.nn.ctc_beam_search_decoder(logits, inputs.shape[2] // 8, merge_repeated = False);
    tokens = tf.cast(tf.sparse.to_dense(decoded[0]), dtype = tf.int64);
    return self.tokenizer.translate(tokens);

if __name__ == "__main__":

  if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " <image>");
    exit();
  img = cv2.imread(sys.argv[1]);
  if img is None:
    print('failed to open image!');
    exit();
  text_recognizer = TextRecognizer();
  print(text_recognizer.recognize(img));
