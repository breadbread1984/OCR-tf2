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
    if width > new_width:
      img = cv2.resize(img, (new_width, height));
    elif width < new_width:
      padding = 255 * np.ones((height, new_width - width, 3), dtype = np.uint8);
      left_width = np.random.randint(low = 0, high = padding.shape[1]);
      left_padding = padding[:,:left_width,:];
      right_padding = padding[:,left_width:,:];
      img = np.concatenate([left_padding, img, right_padding], axis = 1);
    return img;

  def recognize(self, img, preprocess = True):

    if preprocess == True:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
      img = self.resize(img);
      img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY);
      img = tf.reshape(img, (1,img.shape[0],img.shape[1],1));
      inputs = tf.cast(img, dtype = tf.float32) / 255.; # input.shape = (1, seq_length, 32);
    else:
      inputs = img;
    logits = self.ocr(inputs);
    logits = tf.transpose(logits, (1,0,2)); # logits.shape = (seq_length, batch_size, num_class)
    decoded, _ = tf.nn.ctc_beam_search_decoder(logits, [inputs.shape[2] // 8], beam_width = 4);
    tokens = tf.cast(tf.sparse.to_dense(decoded[0]), dtype = tf.int64);
    return self.tokenizer.translate(tokens[0]), decoded[0];

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
