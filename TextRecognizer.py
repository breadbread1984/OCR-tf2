#!/usr/bin/python3

import sys;
import tensorflow as tf;
from models import OCR;

class TextRecognizer(object):

  def __init__(self, num_class = 100):

    self.ocr = OCR(num_class);
    if exists(join('model', 'ocr.h5')):
      self.ocr = tf.keras.models.load_model(join('model','ocr.h5'), compile = False);

  def detect(self, img):

    img = tf.expand_dims(img, axis = 0); # img.shape = (1, seq_length, 32);
    logits = self.ocr(img);
    decoded, log_pro = tf.nn.ctc_beam_search_decoder(logits, img.shape[1] // 8, merge_repeated = False);
    

if __name__ == "__main__":

  if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " <image>");
    exit();
  img = cv2.imread(sys.argv[1]);
  if img is None:
    print('failed to open image!');
    exit();
  text_recognizer = TextRecognizer();
  
