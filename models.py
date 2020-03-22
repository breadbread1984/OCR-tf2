#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;

def CTPN(hidden_units = 128, output_units = 512):

  # input image must in RGB order
  # 1) feature extraction
  inputs = tf.keras.Input((None, None, 3));
  results = tf.keras.layers.Lambda(lambda x: x - tf.reshape([123.68, 116.78, 103.94], (1,1,1,3)))(inputs);
  vgg16 = tf.keras.applications.VGG16(input_tensor = results, include_top = False, weights = 'imagenet');
  # 2) input layer
  before_reshape = tf.keras.layers.Conv2D(filters = 512, kernel_size = (3,3), padding = 'same')(vgg16.get_layer('block5_conv3').output); # before_reshape.shape = (batch, h, w, c = 512)
  # 3) bidirection LSTM on every line of feature
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[-2], x.shape[-1])))(before_reshape); # results.shape = (batch * h, w, c = 512)
  results = tf.keras.layers.Bidirectional(layer = tf.keras.layers.LSTM(hidden_units, return_sequences = True), 
                                         backward_layer = tf.keras.layers.LSTM(hidden_units, return_sequences = True, go_backwards = True), 
                                         input_shape = (-1, hidden_units),
                                         merge_mode = 'concat')(results);                             # results.shape = (batch * h, w, c = 128 * 2)
  results = tf.keras.layers.Dense(units = output_units)(results);                                     # results.shape = (batch * h, w, c = 512)
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x[0], (-1, tf.shape(x[1])[-3], tf.shape(x[1])[-2], x[0].shape[-1])))([results, before_reshape]); # results.shape = (batch, h, w, c = 512)
  # 4) output layer
  bbox_pred = tf.keras.layers.Dense(units = 10 * 6)(results); # bbox_pred.shape = (batch, h, w, c = 60)
  bbox_pred = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[-3], tf.shape(x)[-2], 10, 6)))(bbox_pred); # bbox_pred.shape = (batch, h, w, anchor_num = 10, 6) in sequence of (dx dy dw dh logits0 logits1)
  
  return tf.keras.Model(inputs = inputs, outputs = bbox_pred);

def Loss(max_fg_anchors = 128, max_bg_anchors = 128, rpn_neg_thres = 0.3, rpn_pos_thres = 0.7):

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
  bbox_pred = tf.keras.Input((None, None, 10, 6), batch_size = 1); # bbox_pred = (1, h, w, anchor_num = 10, 4)
  gt_bbox = tf.keras.Input((None, 4), batch_size = 1); # gt_bbox.shape = (1, n, 4) in sequence of (xmin, ymin, xmax, ymax, class)
  # anchor target layer
  grid = tf.keras.layers.Lambda(lambda x: tf.stack([
    tf.tile(tf.reshape(16 * tf.range(tf.cast(tf.shape(x)[-3], dtype = tf.float32), dtype = tf.float32), (1, tf.shape(x)[-3])), (tf.shape(x)[-4], 1)),
    tf.tile(tf.reshape(16 * tf.range(tf.cast(tf.shape(x)[-4], dtype = tf.float32), dtype = tf.float32), (tf.shape(x)[-4], 1)), (1, tf.shape(x)[-3]))
    ], axis = -1))(bbox_pred); # grid.shape = (h, w, 2) in sequence of (x,y), 16 because vgg16's block5_conv3's output size is 1 / 16 of input image size
  all_anchors = tf.keras.layers.Lambda(lambda x, anchors:
    tf.expand_dims(tf.concat([x,x], axis = -1), axis = -2) + tf.reshape(anchors, (1,1,10,4)), # shape = (1, 1, 10, 4)
    arguments = {'anchors': anchors})(grid); # all_anchors.shape = (h, w, 10, 4) in sequence of (xmin ymin xmax ymax)
  # get overlaps between anchors and groud truths
  all_anchors_reshape = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = -2))(all_anchors); # all_anchors_reshape.shape = (h, w, 10, 1, 4)
  gt = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (1,1,1,-1,4)))(gt_bbox); # gt.shape = (1, 1, 1, m, 5)
  upperleft = tf.keras.layers.Lambda(lambda x: tf.math.maximum(x[0][...,0:2], x[1][...,0:2]))([all_anchors_reshape, gt]); # upperleft.shape = (h, w, 10, m, 2)
  downright = tf.keras.layers.Lambda(lambda x: tf.math.minimum(x[0][...,2:4], x[1][...,2:4]))([all_anchors_reshape, gt]); # downright.shape = (h, w, 10, m, 2)
  intersect_wh = tf.keras.layers.Lambda(lambda x: tf.math.maximum(x[1] - x[0], 0.))([upperleft, downright]); # intersect_wh.shape = (h, w, 10, m, 2)
  intersect_area = tf.keras.layers.Lambda(lambda x: x[...,0] * x[...,1])(intersect_wh); # intersect_area.shape = (h, w, 10, m)
  anchors_wh = tf.keras.layers.Lambda(lambda x: x[...,2:4] - x[...,0:2] + 1)(all_anchors_reshape); # anchors_wh.shape = (h, w, 10, 1, 2)
  anchors_area = tf.keras.layers.Lambda(lambda x: x[...,0] * x[...,1])(anchors_wh); # anchors_area.shape = (h, w, 10, 1)
  gt_bbox_wh = tf.keras.layers.Lambda(lambda x: x[...,2:4] - x[...,0:2] + 1)(gt); # gt_bbox_wh.shape = (1, 1, 1, m, 2)
  gt_bbox_area = tf.keras.layers.Lambda(lambda x: x[...,0] * x[...,1])(gt_bbox_wh); # gt_bbox_area.shape = (1, 1, 1, m)
  iou = tf.keras.layers.Lambda(lambda x: x[0] / (x[1] + x[2] - x[0]))([intersect_area, anchors_area, gt_bbox_area]); # iou.shape = (h, w, 10, m)
  best_iou = tf.keras.layers.Lambda(lambda x: tf.math.reduce_max(x, axis = -1))(iou); # best_iou.shape = (h, w, 10)
  best_gt_idx = tf.keras.layers.Lambda(lambda x: tf.math.argmax(x, axis = -1, output_type = tf.int32))(iou); # best_gt.shape = (h, w, 10)
  # 1) label anchor boxes with 1 when overlap > 0.7, 0 when overlap < 0.3, -1 with others
  labels = tf.keras.layers.Lambda(lambda x, nt, pt: tf.where(tf.math.less(x, nt), tf.zeros_like(x), tf.where(tf.math.greater(x,pt), tf.ones_like(x), -tf.ones_like(x))), 
                                  arguments = {'nt': rpn_neg_thres, 'pt': rpn_pos_thres})(best_iou); # labels.shape = (h, w, 10)
  # when there are too many positives, random drop some
  count = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(tf.where(tf.math.equal(x, 1), tf.ones_like(x), tf.zeros_like(x))))(labels);
  labels = tf.keras.layers.Lambda(lambda x, n: tf.cond(tf.math.less(x[0], n),
                                                       true_fn = lambda: x[1],
                                                       false_fn = lambda: tf.where(tf.math.logical_and(tf.math.equal(x[1], 1), tf.math.greater(tf.random.uniform(tf.shape(x[1])), n / x[0])), -tf.ones_like(x[1]), x[1])), 
                                  arguments = {'n': max_fg_anchors})([count, labels]); # labels.shape = (h, w, 10)
  # when there are too many negatives, random drop some
  count = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(tf.where(tf.math.equal(x, 0), tf.ones_like(x), tf.zeros_like(x))))(labels);
  labels = tf.keras.layers.Lambda(lambda x, n: tf.cond(tf.math.less(x[0], n),
                                                       true_fn = lambda: x[1],
                                                       false_fn = lambda: tf.where(tf.math.logical_and(tf.math.equal(x[1], 0), tf.math.greater(tf.random.uniform(tf.shape(x[1])), n / x[0])), -tf.ones_like(x[1]), x[1])),
                                  arguments = {'n': max_bg_anchors})([count, labels]); # labels.shape = (h, w, 10)
  labels = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = 0))(labels); # labels.shape = (1, h, w, 10)
  # 2) get prediction
  anchors_wh = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis = -2))(anchors_wh); # anchors_wh.shape = (h, w, 10, 2)
  anchors_centers = tf.keras.layers.Lambda(lambda x: x[0][..., 0:2] + x[1] / 2)([all_anchors, anchors_wh]); # anchors_centers.shape = (h, w, 10, 2)
  best_gt = tf.keras.layers.Lambda(lambda x: tf.gather(tf.squeeze(x[1], axis = 0), x[0]))([best_gt_idx, gt_bbox]); # best_gt.shape = (h, w, 10, 5)
  best_gt_wh = tf.keras.layers.Lambda(lambda x: x[..., 2:4] - x[..., 0:2] + 1)(best_gt); # best_gt_wh.shape = (h, w, 10, 2)
  best_gt_centers = tf.keras.layers.Lambda(lambda x: x[0][..., 0:2] + x[1] / 2)([best_gt, best_gt_wh]); # best_gt_centers.shape = (h, w, 10, 2)
  target_dxdy = tf.keras.layers.Lambda(lambda x: (x[0] - x[1]) / x[2])([best_gt_centers, anchors_centers, anchors_wh]); # target_dxdy.shape = (h, w, 10, 2)
  target_dwdh = tf.keras.layers.Lambda(lambda x: x[0] / x[1])([best_gt_wh, anchors_wh]); # target_dwdh.shape = (h, w, 10, 2)
  bbox_target = tf.keras.layers.Concatenate(axis = -1)([target_dxdy, target_dwdh]); # bbox_target.shape = (h, w, 10, 4)
  bbox_target = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = 0))(bbox_target); # bbox_target.shape = (1, h, w, 10, 4)
  # 3) get class loss at locations without -1 label
  mask = tf.keras.layers.Lambda(lambda x: tf.math.not_equal(x, -1))(labels); # mask.shape = (1, h, w, 10)
  cls_loss = tf.keras.layers.Lambda(lambda x: tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)(tf.boolean_mask(x[1], x[0]), tf.boolean_mask(x[2][..., -2:], x[0])))([mask, labels, bbox_pred]);
  # 4) get prediction loss at locations without -1 label
  pred_loss = tf.keras.layers.Lambda(lambda x: tf.keras.losses.MeanAbsoluteError()(tf.boolean_mask(x[1], x[0]), tf.boolean_mask(x[2][..., :4], x[0])))([mask, bbox_target, bbox_pred]);
  loss = tf.keras.layers.Lambda(lambda x: x[0] + x[1])([cls_loss, pred_loss]);
  return tf.keras.Model(inputs = (bbox_pred, gt_bbox), outputs = loss);

if __name__ == "__main__":

  assert tf.executing_eagerly();
  a = tf.constant(np.random.normal(size = (10,256,256,3)))
  ctpn = CTPN();
  ctpn.save('ctpn.h5')
  bbox_pred = ctpn(a)
  loss = Loss();
  b = tf.constant(np.random.normal(size = (1,10,4)), dtype = tf.float32);
  anchors = loss([bbox_pred[0:1,...], b]);
  print(anchors);
  print(bbox_pred)
  
