#!/usr/bin/python3

from os import mkdir;
from os.path import join, exists;
import tensorflow as tf;
from create_dataset import parse_function;
from models import CTPN, Loss;

dataset_size = 3421;

def main():
    
  ctpn = CTPN();
  loss = Loss();
  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps = 60000, decay_rate = 0.5));
  # load dataset
  trainset = tf.data.TFRecordDataset(join('datasets', 'trainset.tfrecord')).repeat(-1).map(parse_function).batch(1).prefetch(tf.data.experimental.AUTOTUNE);
  # restore from existing checkpoint
  if False == exists('checkpoints'): mkdir('checkpoints');
  checkpoint = tf.train.Checkpoint(model = ctpn, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  # create log
  log = tf.summary.create_file_writer('checkpoints');
  # train model
  avg_loss = tf.keras.metrics.Mean(name = "loss", dtype = tf.float32);
  for image, labels in trainset:
    with tf.GradientTape() as tape:
      bbox_pred = ctpn(image);
      l = loss([bbox_pred, labels]);
    avg_loss.update_state(l);
    # write log
    if tf.equal(optimizer.iterations % 100, 0):
      with log.as_default():
        tf.summary.scalar('loss', avg_loss.result(), step = optimizer.iterations);
      print('Step #%d Loss: %.6f lr: %.6f' % (optimizer.iterations, avg_loss.result(), optimizer._hyper['learning_rate'](optimizer.iterations)));
      if avg_loss.result() < 0.01: break;
      avg_loss.reset_states();
    grads = tape.gradient(l, model.trainable_variables);
    optimizer.apply_gradients(zip(grads, model.trainable_variables));
    # save model
    if tf.equal(optimizer.iterations % 100, 0):
      checkpoint.save(join('checkpoints', 'ckpt'));
  # save the network structure with weights
  if False == exists('model'): mkdir('model');
  ctpn.save(join('model','ctpn.h5'));

if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();
