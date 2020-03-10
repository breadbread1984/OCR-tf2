#!/usr/bin/python3

import tensorflow as tf;

def CTPN(input_shape, hidden_units = 128, output_units = 512):

  # input image must in RGB order
  inputs = tf.keras.Input(input_shape[-3:]);
  results = tf.keras.layers.Lambda(lambda x: x - tf.reshape([123.68, 116.78, 103.94], (1,1,1,3)))(inputs);
  vgg16 = tf.keras.applications.VGG16(input_tensor = results, include_top = False, weights = 'imagenet');
  # bidirection LSTM
  before_reshape = tf.keras.layers.Conv2D(filters = 512, kernel_size = (3,3))(vgg16.outputs[0]);
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, x.shape[-2], x.shape[-1])))(before_reshape); # results.shape = (batch * h, w, c = 512)
  results = tf.keras.layers.Bidirectional(layer = tf.keras.layers.LSTM(hidden_units, return_sequences = True), 
                                         backward_layer = tf.keras.layers.LSTM(hidden_units, return_sequences = True, go_backwards = True), 
                                         input_shape = (results.shape[-2], hidden_units),
                                         merge_mode = 'concat')(results);                             # results.shape = (batch * h, w, c = hidden_units * 2)
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, x.shape[-1])))(results);              # results.shape = (batch * h * w, c = hidden_units * 2)
  results = tf.keras.layers.Dense(units = output_units)(results);                                     # results.shape = (batch * h * w, c= output_units)
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x[0], (-1, x[1].shape[-3], x[1].shape[-2], x[0].shape[-1])))([results, before_reshape]);
  return tf.keras.Model(inputs = inputs, outputs = results);

if __name__ == "__main__":

  assert tf.executing_eagerly();
  ctpn = CTPN((512,512,3));
  ctpn.save('ctpn.h5');
  
