#!/usr/bin/python3

import sys;
from os import mkdir;
from os.path import join, exists;
import cv2;
import tensorflow as tf;
from create_dataset import parse_function;
from models import Loss, OCR;
from TextDetector import TextDetector;

dataset_size = 3421;
num_class = 100;

def train_cptn():

  detector = TextDetector();
  loss = Loss();
  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-5, decay_steps = 30000, decay_rate = 0.9));
  # load dataset
  trainset = tf.data.TFRecordDataset(join('datasets', 'trainset.tfrecord')).repeat(-1).map(parse_function).batch(1).prefetch(tf.data.experimental.AUTOTUNE);
  # restore from existing checkpoint
  if False == exists('checkpoints'): mkdir('checkpoints');
  checkpoint = tf.train.Checkpoint(model = detector.ctpn, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  # create log
  log = tf.summary.create_file_writer('checkpoints');
  # train model
  avg_loss = tf.keras.metrics.Mean(name = "loss", dtype = tf.float32);
  for image, labels in trainset:
    if labels.shape[1] == 0:
      print("skip sample without labels");
      continue;
    with tf.GradientTape() as tape:
      bbox_pred = detector.ctpn(image);
      l = loss([bbox_pred, labels]);
    avg_loss.update_state(l);
    # write log
    if tf.equal(optimizer.iterations % 100, 0):
      with log.as_default():
        tf.summary.scalar('loss', avg_loss.result(), step = optimizer.iterations);
        # draw text detection results
        text_lines, _, _ = detector.detect(image, False);
        image = image[0,...].numpy().astype('uint8');
        for text_line in text_lines:
          cv2.rectangle(image, (int(text_line[0]), int(text_line[1])), (int(text_line[2]), int(text_line[3])), (0, 255, 0), 2);
        image = tf.expand_dims(image, axis = 0);
        tf.summary.image('text lines', image, step = optimizer.iterations);
      print('Step #%d Loss: %.6f lr: %.6f' % (optimizer.iterations, avg_loss.result(), optimizer._hyper['learning_rate'](optimizer.iterations)));
      if avg_loss.result() < 0.01: break;
      avg_loss.reset_states();
    grads = tape.gradient(l, detector.ctpn.trainable_variables);
    if tf.reduce_any([tf.reduce_any(tf.math.is_nan(grad)) for grad in grads]) == True:
      print("NaN was detected in gradients, skip gradient apply!");
      continue;
    optimizer.apply_gradients(zip(grads, detector.ctpn.trainable_variables));
    # save model
    if tf.equal(optimizer.iterations % 2000, 0):
      checkpoint.save(join('checkpoints', 'ckpt'));
  # save the network structure with weights
  if False == exists('model'): mkdir('model');
  detector.ctpn.save(join('model','ctpn.h5'));

def train_ocr():

  ocr = OCR(num_class);
  

if __name__ == "__main__":

  assert tf.executing_eagerly();
  if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " (train_cptn|train_lstm)");
    exit(1);
  if sys,argv[1] not in ['train_cptn', 'train_ocr']:
    print("only support train_cptn or train_ocr!");
    exit(1);
  if sys.argv[1] == "train_cptn":
    train_cptn();
  else:
    train_ocr();
