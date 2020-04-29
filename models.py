#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;

def CTPN(hidden_units = 128, output_units = 512):

  # input image must in RGB order
  # 1) feature extraction
  inputs = tf.keras.Input((None, None, 3), batch_size = 1);
  results = tf.keras.layers.Lambda(lambda x: x - tf.reshape([123.68, 116.78, 103.94], (1,1,1,3)))(inputs);
  vgg16 = tf.keras.applications.VGG16(input_tensor = results, include_top = False, weights = 'imagenet');
  # 2) input layer
  before_reshape = tf.keras.layers.Conv2D(filters = 512, kernel_size = (3,3), padding = 'same', kernel_regularizer = tf.keras.regularizers.l2(l = 5e-3))(vgg16.get_layer('block5_conv3').output); # before_reshape.shape = (batch, h, w, c = 512)
  # 3) bidirection LSTM on every line of feature
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[-2], x.shape[-1])))(before_reshape); # results.shape = (batch * h, w, c = 512)
  results = tf.keras.layers.Bidirectional(layer = tf.keras.layers.LSTM(hidden_units, return_sequences = True, kernel_regularizer = tf.keras.regularizers.l2(l = 5e-3)), 
                                         backward_layer = tf.keras.layers.LSTM(hidden_units, return_sequences = True, go_backwards = True, kernel_regularizer = tf.keras.regularizers.l2(l = 5e-3)), 
                                         input_shape = (-1, hidden_units),
                                         merge_mode = 'concat')(results);                             # results.shape = (batch * h, w, c = 128 * 2)
  results = tf.keras.layers.Dense(units = output_units, kernel_regularizer = tf.keras.regularizers.l2(l = 5e-3))(results);                                     # results.shape = (batch * h, w, c = 512)
  results = tf.keras.layers.Lambda(lambda x: tf.reshape(x[0], (-1, tf.shape(x[1])[-3], tf.shape(x[1])[-2], x[0].shape[-1])))([results, before_reshape]); # results.shape = (batch, h, w, c = 512)
  # 4) output layer
  bbox_pred = tf.keras.layers.Dense(units = 10 * 6, kernel_regularizer = tf.keras.regularizers.l2(l = 5e-3))(results); # bbox_pred.shape = (batch, h, w, c = 60)
  bbox_pred = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[-3], tf.shape(x)[-2], 10, 6)))(bbox_pred); # bbox_pred.shape = (batch, h, w, anchor_num = 10, 6) in sequence of (dx dy dw dh logits0 logits1)
  
  return tf.keras.Model(inputs = inputs, outputs = bbox_pred);

def OutputParser(min_size = 8, pre_nms_topn = 12000, post_nms_topn = 1000, nms_thres = 0.7):
    
  # constant anchors
  # anchor corner coordinates with respect to upper left corner of every grid
  hws = [(11,16),(16,16),(23,16),(33,16),(48,16),(68,16),(97,16),(139,16),(198,16),(283,16)]; # (h,w)
  anchors = list();
  for hw in hws:
    h, w = hw;
    x_ctr = (0 + 15) * 0.5;
    y_ctr = (0 + 15) * 0.5;
    scaled_anchor = (x_ctr - w / 2, y_ctr - h / 2, x_ctr + w / 2, y_ctr + h / 2); # xmin ymin xmax ymax
    anchors.append(scaled_anchor);
  anchors = np.array(anchors, dtype = np.float32); # anchors.shape = (10, 4)

  bbox_pred = tf.keras.Input((None, None, 10, 6), batch_size = 1); # bbox_pred.shape = (h, w, 10, 6)
  # anchor target layer
  grid = tf.keras.layers.Lambda(lambda x: tf.stack([
    tf.tile(tf.reshape(16 * tf.range(tf.cast(tf.shape(x)[-3], dtype = tf.float32), dtype = tf.float32), (1, tf.shape(x)[-3])), (tf.shape(x)[-4], 1)),
    tf.tile(tf.reshape(16 * tf.range(tf.cast(tf.shape(x)[-4], dtype = tf.float32), dtype = tf.float32), (tf.shape(x)[-4], 1)), (1, tf.shape(x)[-3]))
    ], axis = -1))(bbox_pred); # grid.shape = (h, w, 2) in sequence of (x,y), 16 because vgg16's block5_conv3's output size is 1 / 16 of input image size
  all_anchors = tf.keras.layers.Lambda(lambda x, anchors:
    tf.expand_dims(tf.concat([x,x], axis = -1), axis = -2) + tf.reshape(anchors, (1,1,10,4)), # shape = (1, 1, 10, 4)
    arguments = {'anchors': anchors})(grid); # all_anchors.shape = (h, w, 10, 4) in sequence of (xmin ymin xmax ymax)
  scores = tf.keras.layers.Lambda(lambda x: tf.keras.layers.Softmax()(x[0,...,-2:])[...,1:2])(bbox_pred); # score.shape = (h, w, 10, 1)
  # transform relative to absolute representation
  anchors_wh = tf.keras.layers.Lambda(lambda x: x[...,2:4] - x[...,0:2] + 1)(all_anchors); # anchors_wh.shape = (h, w, 10, 2)
  anchors_centers = tf.keras.layers.Lambda(lambda x: x[0][..., 0:2] + x[1] / 2)([all_anchors, anchors_wh]); # anchors_centers.shape = (h, w, 10, 2)
  target_centers = tf.keras.layers.Lambda(lambda x: x[0][0,...,:2] * x[1] + x[2])([bbox_pred, anchors_wh, anchors_centers]); # target_centers.shape = (h, w, 10, 2)
  target_wh = tf.keras.layers.Lambda(lambda x: tf.math.exp(x[0][0,...,2:4]) * x[1])([bbox_pred, anchors_wh]); # target_wh.shape = (h, w, 10, 2)
  upperleft = tf.keras.layers.Lambda(lambda x: x[0] - x[1] / 2)([target_centers, target_wh]); # upperleft.shape = (h, w, 10, 2)
  downright = tf.keras.layers.Lambda(lambda x: x[0] + x[1] / 2)([target_centers, target_wh]); # downright.shape = (h, w, 10, 2)
  bbox = tf.keras.layers.Concatenate(axis = -1)([upperleft, downright]); # bbox.shape = (h, w, 10, 4) in sequence of (xmin ymin xmax ymax)
  # clip boxes to make all outputs within the border
  bbox = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, [0,0,0,0], 16 * tf.cast([tf.shape(x)[-3],tf.shape(x)[-4],tf.shape(x)[-3],tf.shape(x)[-4]], dtype = tf.float32) - 1))(bbox);
  bbox = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, 4)))(bbox); # bbox.shape = (-1, 4)
  scores = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, 1)))(scores); # scores.shape = (-1, 1)
  # filter boxes
  bbox_wh = tf.keras.layers.Lambda(lambda x: x[..., 2:4] - x[..., 0:2] + 1)(bbox); # bbox_wh.shape = (-1, 2)
  mask = tf.keras.layers.Lambda(lambda x, m: tf.math.logical_and(tf.math.greater_equal(x[...,0], m), tf.math.greater_equal(x[...,1], m)), arguments = {'m': min_size})(bbox_wh); # mask.shape = (n)
  filtered_bbox = tf.keras.layers.Lambda(lambda x: tf.boolean_mask(x[0], x[1]))([bbox, mask]); # filtered_bbox.shape = (n, 4)
  filtered_scores = tf.keras.layers.Lambda(lambda x: tf.boolean_mask(x[0], x[1]))([scores, mask]); # filtered_scores.shape = (n, 1)
  # nms
  idx = tf.keras.layers.Lambda(lambda x: tf.argsort(x, axis = 0, direction = 'DESCENDING'))(filtered_scores); # idx.shape = (n, 1)
  sorted_bbox = tf.keras.layers.Lambda(lambda x,n : tf.gather_nd(x[0], x[1])[:n,...], arguments = {'n': pre_nms_topn})([filtered_bbox, idx]); # sorted_bbox.shape = (n, 4)
  sorted_scores = tf.keras.layers.Lambda(lambda x, n: tf.gather_nd(x[0], x[1])[:n,...], arguments = {'n': pre_nms_topn})([filtered_scores, idx]); # sorted_scores.shape = (n, 1)
  def condition(index, bbox, scores):
    return index < tf.shape(bbox)[0];
  def body(index, bbox, scores):
    current_bbox = tf.expand_dims(bbox[index:index + 1,...], axis = 1); # current_bbox.shape = (1, 1, 4)
    following_bbox = tf.expand_dims(bbox[index + 1:,...], axis = 0); # following_bbox.shape = (1, m, 4)
    following_scores = tf.expand_dims(scores[index + 1:,...], axis = 0); # following_scores.shape = (1, m, 1)
    upperleft = tf.math.maximum(current_bbox[...,:2], following_bbox[...,:2]); # upperleft.shape = (1, m, 2)
    downright = tf.math.minimum(current_bbox[...,2:], following_bbox[...,2:]); # downright.shape = (1, m, 2)
    intersect_wh = tf.math.maximum(downright - upperleft + 1, 0.); # intersect_wh.shape = (1, m, 2)
    intersect_area = intersect_wh[...,0] * intersect_wh[...,1]; # intersect_area.shape = (1, m)
    current_wh = tf.math.maximum(current_bbox[...,2:] - current_bbox[...,:2] + 1, 0.); # current_wh.shape = (1, 1, 2)
    current_area = current_wh[...,0] * current_wh[...,1]; # current_area.shape = (1, 1)
    following_wh = tf.math.maximum(following_bbox[...,2:] - following_bbox[...,:2] + 1, 0.); # following_wh.shape = (1, m, 2)
    following_area = following_wh[...,0] * following_wh[...,1]; # following_area.shape = (1, m)
    iou = intersect_area / (current_area + following_area - intersect_area); # iou.shape = (1, m)
    mask = tf.where(tf.math.less(iou, nms_thres)); # mask.shape = (1, m)
    filtered_following_bbox = tf.gather_nd(following_bbox, mask); # filtered_following_bbox.shape = (n, 4)
    filtered_following_scores = tf.gather_nd(following_scores, mask); # filtered_following_scores.shape = (n, 1)
    bbox = tf.concat([bbox[:index + 1,...], filtered_following_bbox], axis = 0); # bbox.shape = (m', 4)
    scores = tf.concat([scores[:index+1,...], filtered_following_scores], axis = 0); # scores.shape = (m', 1)
    index += 1;
    return index, bbox, scores;
  _, nms_bbox, nms_scores = tf.keras.layers.Lambda(lambda x: tf.while_loop(condition, body, loop_vars = [tf.constant(0), x[0], x[1]], shape_invariants = [tf.TensorShape([]), tf.TensorShape([None, 4]), tf.TensorShape([None, 1])]))([sorted_bbox, sorted_scores]);
  nms_bbox = tf.keras.layers.Lambda(lambda x, n: x[:n, ...], arguments = {'n': post_nms_topn})(nms_bbox);
  nms_scores = tf.keras.layers.Lambda(lambda x, n: x[:n, ...], arguments = {'n': post_nms_topn})(nms_scores);
  return tf.keras.Model(inputs = bbox_pred, outputs = (nms_bbox, nms_scores));

def GraphBuilder(min_score = 0.7, nms_thres = 0.2, max_horizontal_gap = 50, min_v_overlap = 0.7, min_size_sim = 0.7):

  bbox = tf.keras.Input((4,)); # bbox.shape = (n, 4)
  scores = tf.keras.Input((1,)); # scores.shape = (n, 1)
  # filter proposals below threshold
  mask = tf.keras.layers.Lambda(lambda x, m: tf.where(tf.math.greater(tf.squeeze(x), m)), arguments = {'m': min_score})(scores); # mask.shape = (n)
  filtered_bbox = tf.keras.layers.Lambda(lambda x: tf.gather_nd(x[0], x[1]))([bbox, mask]); # filtered_bbox.shape = (m, 4)
  filtered_scores = tf.keras.layers.Lambda(lambda x: tf.gather_nd(x[0], x[1]))([scores, mask]); # filtered_scores.shape = (m, 1)
  # nms
  idx = tf.keras.layers.Lambda(lambda x: tf.argsort(x, axis = 0, direction = 'DESCENDING'))(filtered_scores); # idx.shape = (m, 1)
  sorted_bbox = tf.keras.layers.Lambda(lambda x: tf.gather_nd(x[0], x[1]))([filtered_bbox, idx]); # sorted_bbox.shape = (m, 4)
  sorted_scores = tf.keras.layers.Lambda(lambda x: tf.gather_nd(x[0], x[1]))([filtered_scores, idx]); # sorted_scores.shape = (m, 1)
  def condition(index, bbox, scores):
    return index < tf.shape(bbox)[0];
  def body(index, bbox, scores):
    current_bbox = tf.expand_dims(bbox[index:index + 1,...], axis = 1); # current_bbox.shape = (1, 1, 4)
    following_bbox = tf.expand_dims(bbox[index + 1:,...], axis = 0); # following_bbox.shape = (1, m, 4)
    following_scores = tf.expand_dims(scores[index + 1:,...], axis = 0); # following_scores.shape = (1, m, 1)
    upperleft = tf.math.maximum(current_bbox[...,:2], following_bbox[...,:2]); # upperleft.shape = (1, m, 2)
    downright = tf.math.minimum(current_bbox[...,2:], following_bbox[...,2:]); # downright.shape = (1, m, 2)
    intersect_wh = tf.math.maximum(downright - upperleft + 1, 0.); # intersect_wh.shape = (1, m, 2)
    intersect_area = intersect_wh[...,0] * intersect_wh[...,1]; # intersect_area.shape = (1, m)
    current_wh = tf.math.maximum(current_bbox[...,2:] - current_bbox[...,:2] + 1, 0.); # current_wh.shape = (1, 1, 2)
    current_area = current_wh[...,0] * current_wh[...,1]; # current_area.shape = (1, 1)
    following_wh = tf.math.maximum(following_bbox[...,2:] - following_bbox[...,:2] + 1, 0.); # following_wh.shape = (1, m, 2)
    following_area = following_wh[...,0] * following_wh[...,1]; # following_area.shape = (1, m)
    iou = intersect_area / (current_area + following_area - intersect_area); # iou.shape = (1, m)
    mask = tf.where(tf.math.less(iou, nms_thres)); # mask.shape = (1, m)
    filtered_following_bbox = tf.gather_nd(following_bbox, mask); # filtered_following_bbox.shape = (n, 4)
    filtered_following_scores = tf.gather_nd(following_scores, mask); # filtered_following_scores.shape = (n, 1)
    bbox = tf.concat([bbox[:index + 1,...], filtered_following_bbox], axis = 0); # bbox.shape = (m', 4)
    scores = tf.concat([scores[:index+1,...], filtered_following_scores], axis = 0); # scores.shape = (m', 1)
    index += 1;
    return index, bbox, scores;
  _, nms_bbox, nms_scores = tf.keras.layers.Lambda(lambda x: tf.while_loop(condition, body, loop_vars = [tf.constant(0), x[0], x[1]], shape_invariants = [tf.TensorShape([]), tf.TensorShape([None, 4]), tf.TensorShape([None, 1])]))([sorted_bbox, sorted_scores]);
  # construct graph
  minx_diff = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[..., 0], axis = 1) - tf.expand_dims(x[..., 0], axis = 0))(nms_bbox); # successor_mask.shape = (m',m')
  # overlap
  upperleft_h = tf.keras.layers.Lambda(lambda x: tf.math.maximum(tf.expand_dims(x[...,1], axis = 0), tf.expand_dims(x[...,1], axis = 1)))(nms_bbox); # upperleft.shape = (m',m')
  downright_h = tf.keras.layers.Lambda(lambda x: tf.math.minimum(tf.expand_dims(x[...,3], axis = 0), tf.expand_dims(x[...,3], axis = 1)))(nms_bbox); # downright.shape = (m',m')
  intersect_h = tf.keras.layers.Lambda(lambda x: tf.math.maximum(x[1] - x[0] + 1, 0.))([upperleft_h, downright_h]); # intersect_wh.shape = (m',m')
  bbox_h = tf.keras.layers.Lambda(lambda x: x[...,3] - x[...,1])(nms_bbox); # bbox_h.shape = (m')
  max_h = tf.keras.layers.Lambda(lambda x: tf.math.maximum(tf.expand_dims(x, axis = 0), tf.expand_dims(x, axis = 1)))(bbox_h); # max_h.shape = (m',m')
  min_h = tf.keras.layers.Lambda(lambda x: tf.math.minimum(tf.expand_dims(x, axis = 0), tf.expand_dims(x, axis = 1)))(bbox_h); # min_h.shape = (m',m')
  overlap_h = tf.keras.layers.Lambda(lambda x: x[0] / x[1])([intersect_h, min_h]); # overlap.shape = (m',m')
  # size similarity
  size_similarity = tf.keras.layers.Lambda(lambda x: x[0] / x[1])([min_h, max_h]); # size_similarity.shape = (m',m')
  # successor and precursor mask
  is_successor = tf.keras.layers.Lambda(lambda x, g, t1, t2:
    tf.math.logical_and(
      tf.math.logical_and(tf.math.greater_equal(x[0], 1), tf.math.less_equal(x[0], g)),
      tf.math.logical_and(tf.math.greater_equal(x[1], t1), tf.math.greater_equal(x[2], t2))
    ),
    arguments = {'g': max_horizontal_gap, 't1': min_v_overlap, 't2': min_size_sim}
  )([minx_diff,overlap_h,size_similarity]); # is_successor.shape = (m',m') row is successor of col
  is_precursor = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (1,0)))(is_successor); # is_precursor.shape = (m',m') row is precursor of col
  row_multiplied_scores = tf.keras.layers.Lambda(lambda x: tf.tile(tf.transpose(x, (1, 0)),(tf.shape(x)[0], 1)))(nms_scores);
  def max_by_row(x):
    mask = x[0];
    scores = x[1];
    masked_scores = tf.where(mask, scores, tf.zeros_like(scores));
    max_mask = tf.cond(tf.math.reduce_any(mask), true_fn = lambda: tf.one_hot(tf.math.argmax(masked_scores), tf.shape(mask)[0]), false_fn = lambda: tf.zeros_like(mask, dtype = tf.float32));
    max_mask = tf.cast(max_mask, dtype = tf.bool);
    return max_mask;
  max_precursor_mask = tf.keras.layers.Lambda(lambda x: tf.map_fn(max_by_row, (x[0], x[1]), dtype = tf.bool))([is_successor, row_multiplied_scores]);
  max_successor_mask = tf.keras.layers.Lambda(lambda x: tf.map_fn(max_by_row, (x[0], x[1]), dtype = tf.bool))([is_precursor, row_multiplied_scores]);
  graph = tf.keras.layers.Lambda(lambda x: tf.math.logical_and(x[0], tf.transpose(x[1], (1, 0))))([max_successor_mask, max_precursor_mask]);
  return tf.keras.Model(inputs = (bbox, scores), outputs = (graph, nms_bbox, nms_scores));

def Loss(max_fg_anchors = 128, max_bg_anchors = 128, rpn_neg_thres = 0.3, rpn_pos_thres = 0.7):

  # constant anchors
  # anchor corner coordinates with respect to upper left corner of every grid
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
  gt = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (1,1,1,-1,4)))(gt_bbox); # gt.shape = (1, 1, 1, m, 4)
  upperleft = tf.keras.layers.Lambda(lambda x: tf.math.maximum(x[0][...,0:2], x[1][...,0:2]))([all_anchors_reshape, gt]); # upperleft.shape = (h, w, 10, m, 2)
  downright = tf.keras.layers.Lambda(lambda x: tf.math.minimum(x[0][...,2:4], x[1][...,2:4]))([all_anchors_reshape, gt]); # downright.shape = (h, w, 10, m, 2)
  intersect_wh = tf.keras.layers.Lambda(lambda x: tf.math.maximum(x[1] - x[0] + 1, 0.))([upperleft, downright]); # intersect_wh.shape = (h, w, 10, m, 2)
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
  # labels of anchors touching the borders are set to -1
  labels = tf.keras.layers.Lambda(lambda x: tf.where(
    tf.math.logical_and(
      tf.math.logical_and(
        tf.math.logical_and(tf.math.greater_equal(x[1][...,0], 0), tf.math.less(x[1][...,0], tf.cast(16 * tf.shape(x[1])[-3], dtype = tf.float32))),
        tf.math.logical_and(tf.math.greater_equal(x[1][...,1], 0), tf.math.less(x[1][...,1], tf.cast(16 * tf.shape(x[1])[-4], dtype = tf.float32)))
      ),
      tf.math.logical_and(
        tf.math.logical_and(tf.math.greater_equal(x[1][...,2], 0), tf.math.less(x[1][...,2], tf.cast(16 * tf.shape(x[1])[-3], dtype = tf.float32))),
        tf.math.logical_and(tf.math.greater_equal(x[1][...,3], 0), tf.math.less(x[1][...,3], tf.cast(16 * tf.shape(x[1])[-4], dtype = tf.float32)))
      )
    ),
    x[0], -tf.ones_like(x[0])
  ))([labels, all_anchors]);
  labels = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = 0))(labels); # labels.shape = (1, h, w, 10)
  # 2) get prediction
  anchors_wh = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis = -2))(anchors_wh); # anchors_wh.shape = (h, w, 10, 2)
  anchors_centers = tf.keras.layers.Lambda(lambda x: x[0][..., 0:2] + x[1] / 2)([all_anchors, anchors_wh]); # anchors_centers.shape = (h, w, 10, 2)
  best_gt = tf.keras.layers.Lambda(lambda x: tf.gather(tf.squeeze(x[1], axis = 0), x[0]))([best_gt_idx, gt_bbox]); # best_gt.shape = (h, w, 10, 4)
  best_gt_wh = tf.keras.layers.Lambda(lambda x: x[..., 2:4] - x[..., 0:2] + 1)(best_gt); # best_gt_wh.shape = (h, w, 10, 2)
  best_gt_centers = tf.keras.layers.Lambda(lambda x: x[0][..., 0:2] + x[1] / 2)([best_gt, best_gt_wh]); # best_gt_centers.shape = (h, w, 10, 2)
  # transform absolute to relative represention
  target_dxdy = tf.keras.layers.Lambda(lambda x: (x[0] - x[1]) / x[2])([best_gt_centers, anchors_centers, anchors_wh]); # target_dxdy.shape = (h, w, 10, 2)
  target_dwdh = tf.keras.layers.Lambda(lambda x: tf.math.log(x[0] / x[1]))([best_gt_wh, anchors_wh]); # target_dwdh.shape = (h, w, 10, 2)
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
  parser = OutputParser();
  b, s = parser(bbox_pred);
  connector = Connector();
  o = connector([b,s]);
  print(o);
  loss = Loss();
  b = tf.constant(np.random.normal(size = (1,10,4)), dtype = tf.float32);
  l = loss([bbox_pred[0:1,...], b]);
