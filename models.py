#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;

def CTPN(input_shape, hidden_units = 128, output_units = 512):

  # input image must in RGB order
  # 1) feature extraction
  inputs = tf.keras.Input(input_shape[-3:]);
  results = tf.keras.layers.Lambda(lambda x: x - tf.reshape([123.68, 116.78, 103.94], (1,1,1,3)))(inputs);
  vgg16 = tf.keras.applications.VGG16(input_tensor = results, include_top = False, weights = 'imagenet');
  # 2) input layer
  before_reshape = tf.keras.layers.Conv2D(filters = 512, kernel_size = (3,3))(vgg16.get_layer('block5_conv3').output);
  # 3) bidirection LSTM
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, x.shape[-2], x.shape[-1])))(before_reshape); # results.shape = (batch * h, w, c = 512)
  results = tf.keras.layers.Bidirectional(layer = tf.keras.layers.LSTM(hidden_units, return_sequences = True), 
                                         backward_layer = tf.keras.layers.LSTM(hidden_units, return_sequences = True, go_backwards = True), 
                                         input_shape = (results.shape[-2], hidden_units),
                                         merge_mode = 'concat')(results);                             # results.shape = (batch * h, w, c = hidden_units * 2)
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, x.shape[-1])))(results);              # results.shape = (batch * h * w, c = hidden_units * 2)
  results = tf.keras.layers.Dense(units = output_units)(results);                                     # results.shape = (batch * h * w, c = output_units)
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x[0], (-1, x[1].shape[-3], x[1].shape[-2], x[0].shape[-1])))([results, before_reshape]); # results.shape = (batch, h, w, c = output_units)
  # 4) output layer
  bbox_pred = tf.keras.layers.Dense(units = 10 * 4)(results); # bbox_pred.shape = (batch, h, w, c = 40)
  cls_pred = tf.keras.layers.Dense(units = 10 * 2)(results);  # cls_pred.shape = (batch, h, w, c = 20)
  cls_pred_reshape = tf.keras.layers.Reshape((cls_pred.shape[1], cls_pred.shape[2], -1, 2))(cls_pred); # cls_pred_reshape.shape = (batch, h, w * 10, 2)
  cls_prob = tf.keras.layers.Softmax()(cls_pred_reshape);
  
  return tf.keras.Model(inputs = inputs, outputs = (bbox_pred, cls_pred, cls_prob));

def Loss(img_shape, feat_shape):

  # constant anchors
  hws = [(11,16),(16,16),(23,16),(33,16),(48,16),(68,16),(97,16),(139,16),(198,16),(283,16)]; # (h,w)
  anchors = list();
  for hw in hws:
    h, w = hw;
    x_ctr = (0 + 15) * 0.5;
    y_ctr = (0 + 15) * 0.5;
    scaled_anchor = (x_ctr - w / 2, y_ctr - h / 2, x_ctr + w / 2, y_ctr + h / 2); # xmin ymin xmax ymax
    anchors.append(scaled_anchor);
  anchors = np.array(anchors, dtype = np.float32); # anchors.shape = (10, 4)
  
  # graph defined from here
  bbox_pred = tf.keras.Input((feat_shape[-3], feat_shape[-2], 40));
  cls_pred = tf.keras.Input((feat_shape[-3], feat_shape[-2], 20));
  gt_bbox = tf.keras.Input((5,)); # gt_bbox.shape = (n, 5) in sequence of (xmin, ymin, xmax, ymax, class)
  # anchor target layer
  grid = tf.keras.layers.Lambda(lambda x: tf.stack([
    tf.tile(tf.reshape(16 * tf.range(tf.cast(x.shape[2], dtype = tf.float32), dtype = tf.float32), (1, x.shape[2])), (x.shape[1], 1)),
    tf.tile(tf.reshape(16 * tf.range(tf.cast(x.shape[1], dtype = tf.float32), dtype = tf.float32), (x.shape[1], 1)), (1, x.shape[2]))
    ], axis = -1))(bbox_pred); # grid.shape = (h, w, 2) in sequence of (x,y), 16 because vgg16's block5_conv3's output size is 1 / 16 of input image size
  all_anchors = tf.keras.layers.Lambda(lambda x, anchors:
    tf.expand_dims(tf.concat([x,x], axis = -1), axis = -2) + tf.reshape(anchors, (1,1,10,4)), # shape = (1, 1, 10, 4)
    arguments = {'anchors': anchors})(grid); # all_anchors.shape = (h, w, 10, 4) in sequence of (xmin ymin xmax ymax)
  all_anchors = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, 4)))(all_anchors); # all_anchors = (h * w * 10, 4)
  # filter anchors which crosses over with image borders
  anchors = tf.keras.layers.Lambda(lambda x, h, w: tf.gather(x, tf.reshape(tf.where(
    tf.math.logical_and(
      tf.math.logical_and(
        tf.math.logical_and(tf.math.greater_equal(x[...,0], 0), tf.math.less(x[...,0], w)),
        tf.math.logical_and(tf.math.greater_equal(x[...,1], 0), tf.math.less(x[...,1], h))
      ),
      tf.math.logical_and(
        tf.math.logical_and(tf.math.greater_equal(x[...,2], 0), tf.math.less(x[...,2], w)),
        tf.math.logical_and(tf.math.greater_equal(x[...,3], 0), tf.math.less(x[...,3], h))
      ),
    )), (-1,))), arguments = {'h': img_shape[-3], 'w': img_shape[-2]})(all_anchors); # anchors = (n, 4) in sequence of (xmin ymin xmax ymax)
  # bbox overlap
  anchors = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1,1,4)))(anchors); # anchors.shape = (n, 1, 4)
  gt = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (1,-1,5)))(gt_bbox); # gt_bbox.shape = (1, m. 5)
  upperleft = tf.keras.layers.Lambda(lambda x: tf.math.maximum(x[0][...,0:2], x[1][...,0:2]))([anchors, gt]); # upperleft.shape = (n,m,2)
  downright = tf.keras.layers.Lambda(lambda x: tf.math.minimum(x[0][...,2:4], x[1][...,2:4]))([anchors, gt]); # downright.shape = (n,m,2)
  intersect_wh = tf.keras.layers.Lambda(lambda x: tf.math.maximum(x[1] - x[0], 0.))([upperleft, downright]); # intersect_wh.shape = (n,m,2)
  intersect_area = tf.keras.layers.Lambda(lambda x: x[...,0] * x[...,1])(intersect_wh); # intersect_area.shape = (n,m)
  anchors_wh = tf.keras.layers.Lambda(lambda x: x[...,2:4] - x[...,0:2])(anchors); # anchors_wh.shape = (n, 1, 2)
  anchors_area = tf.keras.layers.Lambda(lambda x: x[...,0] * x[...,1])(anchors_wh); # anchors_area.shape = (n, 1)
  gt_bbox_wh = tf.keras.layers.Lambda(lambda x: x[...,2:4] - x[...,0:2])(gt); # gt_bbox_wh.shape = (1, m, 2)
  gt_bbox_area = tf.keras.layers.Lambda(lambda x: x[...,0] * x[...,1])(gt_bbox_wh); # gt_bbox_area.shape = (1, m)
  iou = tf.keras.layers.Lambda(lambda x: x[0] / (x[1] + x[2] - x[0]))([intersect_area, anchors_area, gt_bbox_area]); # iou.shape = (n, m)
  best_anchor = tf.keras.layers.Lambda(lambda x: tf.math.argmax(iou, axis = 0, output_type = tf.int32))(iou); # best_anchor.shape = (m)
  return tf.keras.Model(inputs = (bbox_pred, cls_pred, gt_bbox), outputs = best_anchor);

if __name__ == "__main__":

  assert tf.executing_eagerly();
  a = tf.constant(np.random.normal(size = (10,256,256,3)))
  ctpn = CTPN((256,256,3));
  bbox_pred, cls_pred, cls_prob = ctpn(a)
  loss = Loss((256,256,3), bbox_pred.shape);
  b = tf.constant(np.random.normal(size = (10,5)), dtype = tf.float32);
  anchors = loss([bbox_pred, cls_pred, b]);
  print(anchors);
  
