#!/usr/bin/python3

import sys;
from os import mkdir;
from os.path import join, exists;
import cv2;
import tensorflow as tf;
from create_dataset import ctpn_parse_function, ocr_parse_function, SampleGenerator;
from models import Loss;
from TextDetector import TextDetector;
from TextRecognizer import TextRecognizer;

dataset_size = 3421;
batch_size = 128;

def train_cptn():

  detector = TextDetector();
  loss = Loss();
  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-5, decay_steps = 30000, decay_rate = 0.9));
  # load dataset
  trainset = tf.data.TFRecordDataset(join('datasets', 'trainset.tfrecord')).repeat(-1).map(ctpn_parse_function).batch(1).prefetch(tf.data.experimental.AUTOTUNE);
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

def to_sparse(labels):

  b = tf.tile(tf.reshape(tf.range(labels.shape[0]), (-1,1,1)),(1,labels.shape[1],1));
  l = tf.tile(tf.reshape(tf.range(labels.shape[1]), (1,-1,1)),(labels.shape[0],1,1));
  indices = tf.cast(tf.reshape(tf.concat([b,l], axis = -1), (-1,2)), dtype = tf.int64);
  values = tf.reshape(labels, (-1,));
  shape = labels.shape;
  return tf.sparse.SparseTensor(indices, values, shape);

def train_ocr():

  generator = SampleGenerator(10);
  recognizer = TextRecognizer();
  optimizer = tf.keras.optimizers.Adam(1e-7);
  # load dataset
  trainset = tf.data.Dataset.from_generator(generator.gen, (tf.float32, tf.int64), (tf.TensorShape([32, None, 3]), tf.TensorShape([None,]))).repeat(-1).map(ocr_parse_function).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  # restore from existing checkpoint
  if False == exists('checkpoints'): mkdir('checkpoints');
  checkpoint = tf.train.Checkpoint(model = recognizer.crnn, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  # create log
  log = tf.summary.create_file_writer('checkpoints');
  # train model
  avg_loss = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32);
  for image, labels in trainset:
    if True == tf.math.reduce_any(tf.math.is_nan(image)):
      print("nan was detected in image! skip current iteration");
      continue;
    with tf.GradientTape() as tape:
      # image.shape = (batch, seq_length, 32)
      pred = recognizer.crnn(image); # logits.shape = (batch, seq_length / 4, class_num + 1)
      if True == tf.math.reduce_any(tf.math.is_nan(pred)):
        print('nan was detected in pred! skip current iteration');
        continue;
      loss = tf.keras.backend.ctc_batch_cost(y_true = labels, y_pred = pred, input_length = tf.tile([[pred.shape[1]]], (batch_size, 1)), label_length = tf.tile([[labels.shape[1]]], (batch_size, 1)));
    avg_loss.update_state(loss);
    # write log
    if tf.equal(optimizer.iterations % 100, 0):
      with log.as_default():
        tf.summary.scalar('loss', avg_loss.result(), step = optimizer.iterations);
        text, decoded = recognizer.recognize(image[0:1,...], False);
        #err = tf.reduce_mean(tf.edit_distance(tf.cast(decoded, tf.int32), to_sparse(tf.cast(labels, dtype = tf.int32))));
        tf.summary.image('image', tf.cast((image[0:1,...] / 2 + 0.5) * 255., dtype = tf.uint8), step = optimizer.iterations);
        tf.summary.text('text', text, step = optimizer.iterations);
        #tf.summary.scalar('word error', err, step = optimizer.iterations);
      print('Step #%d Loss: %.6f' % (optimizer.iterations, avg_loss.result()));
      if avg_loss.result() < 0.01: break;
      avg_loss.reset_states();
    grads = tape.gradient(loss, recognizer.crnn.trainable_variables);
    optimizer.apply_gradients(zip(grads, recognizer.crnn.trainable_variables));
    # save model
    if tf.equal(optimizer.iterations % 2000, 0):
      checkpoint.save(join('checkpoints', 'ckpt'));
  # save the network structure with weights
  if False == exists('model'): mkdir('model');
  recognizer.crnn.save(join('model', 'crnn.h5'));

if __name__ == "__main__":

  assert tf.executing_eagerly();
  if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " (ctpn|ocr)");
    exit(1);
  if sys.argv[1] not in ['ctpn', 'ocr']:
    print("only support ctpn or ocr!");
    exit(1);
  if sys.argv[1] == "ctpn":
    train_cptn();
  else:
    train_ocr();
