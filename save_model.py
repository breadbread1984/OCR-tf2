#!/usr/bin/python3

import sys;
from os import mkdir;
from os.path import join, exists;
import tensorflow as tf;
from models import CTPN;

def main():

  ctpn = CTPN();
  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-5, decay_steps = 30000, decay_rate = 0.1));
  checkpoint = tf.train.Checkpoint(model = ctpn, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  if False == exists("model"): mkdir("model");
  ctpn.save(join("model","ctpn.h5"));

if __name__ == "__main__":

  main();
