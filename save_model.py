#!/usr/bin/python3

import sys;
from os import mkdir;
from os.path import join, exists;
import tensorflow as tf;
from create_dataset import SampleGenerator;
from models import CTPN, CRNN;

def save_ctpn():

  ctpn = CTPN();
  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-5, decay_steps = 30000, decay_rate = 0.1));
  checkpoint = tf.train.Checkpoint(model = ctpn, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  if False == exists("model"): mkdir("model");
  ctpn.save(join("model","ctpn.h5"));

def save_ocr():

  generator = SampleGenerator(10);
  crnn = CRNN(generator.vocab_size());
  optimizer = tf.keras.optimizers.Adam(1e-4);
  checkpoint = tf.train.Checkpoint(model = crnn, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  if False == exists('model'): mkdir("model");
  crnn.save(join("model", "crnn.h5"));

if __name__ == "__main__":

  assert tf.executing_eagerly();
  if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " (ctpn|ocr)");
    exit(1);
  if sys.argv[1] not in ['ctpn', 'ocr']:
    print("only support ctpn or ocr!");
    exit(1);
  if sys.argv[1] == "ctpn":
    save_cptn();
  else:
    save_ocr();
