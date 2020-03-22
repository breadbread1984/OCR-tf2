#!/usr/bin/python3

import sys;
from os.path import exists, join;
import tensorflow as tf;
import cv2;
from models import CTPN;

class TextDetector(object):

  def __init__(self):

    self.ctpn = CTPN();
    if False == exists(join('model', 'ctpn.h5')):
      raise Exception('no model was found under directory model!');
    self.ctpn.load('ctpn.h5');

  def resize(self, img):

    im_size_min = min(img.shape[0:2]);
    im_size_max = max(img.shape[0:2]);
    im_scale = 600 / im_size_min if 600 / float(im_size_min) * im_size_max <= 1200 else 1200 / im_size_max;
    new_h = int(img.shape[0] * im_scale);
    new_w = int(img.shape[1] * im_scale);
    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16;
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16;
    output = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_LINEAR);
    return output, (new_h / img.shape[0], new_w / img.shape[1]);

  def detect(self, img):

    input = cv2.cvtColor(img, cv2.COLOR_BRG2RGB);
    input, scale = self.resize(input);
    inputs = tf.cast(tf.expand_dims(input, axis = 0), dtype = tf.float32); # inputs.shape = (1, h, w, c)
    bbox_pred = self.ctpn(inputs); # bbox_pred = (1, h / 16, w / 16, 10, 6)

if __name__ == "__main__":

  if len(sys.argv) != 2:
    print('Usage: ' + sys.argv[0] + " <image>");
    exit();
  img = cv2.imread(sys.argv[1]);
  if img is None:
    print('failed to open image!');
    exit();
  text_detector = TextDetector();
  text_detector.detect(img);
  
