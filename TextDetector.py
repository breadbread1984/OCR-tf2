#!/usr/bin/python3

import sys;
from os.path import exists, join;
import numpy as np;
import tensorflow as tf;
import cv2;
from models import CTPN, OutputParser, GraphBuilder;

class TextDetector(object):

  def __init__(self):

    self.ctpn = CTPN();
    self.parser = OutputParser();
    self.graph_builder = GraphBuilder();
    if exists(join('model', 'ctpn.h5')):
      self.ctpn = tf.keras.models.load_model(join('model','ctpn.h5'), compile = False);

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

  def subgraph(self, graph):

    # cut a graph into several connected components
    sub_graphs = list();
    for i in range(graph.shape[0]):
      if not tf.math.reduce_any(graph[:, i]) and tf.math.reduce_any(graph[i, :]):
        # find a node with no precursors but has successors, create a connected component from it
        v = i;
        sub_graphs.append([v]);
        # traverse nodes with deep first search
        while tf.math.reduce_any(graph[v, :]):
          v = tf.where(graph[v, :])[0, 0]; # find the first successor
          sub_graphs[-1].append(v); # add the node into subgraph
    return sub_graphs;

  def fit_y(self, X, Y, x1, x2):
    
    # if this group only contains one box
    if tf.math.reduce_sum(tf.cast(tf.math.equal(X, X[0]), dtype = tf.int32)) == X.shape[0]:
      return Y[0], Y[0];
    # else fit with ax+b=y function
    params = np.polyfit(X.numpy(),Y.numpy(), 1)
    linefunc = np.poly1d(params);
    return linefunc(x1), linefunc(x2);

  def detect(self, img, preprocess = True):

    if preprocess == True:
      input = cv2.cvtColor(img, cv2.COLOR_BRG2RGB);
      input, scale = self.resize(input);
      inputs = tf.cast(tf.expand_dims(input, axis = 0), dtype = tf.float32); # inputs.shape = (1, h, w, c)
    else:
      inputs = img;
    bbox_pred = self.ctpn(inputs); # bbox_pred.shape = (1, h / 16, w / 16, 10, 6)
    bbox, bbox_scores = self.parser(bbox_pred); # bbox.shape = (n, 4) bbox_scores.shape = (n, 1)
    graph, nms_bbox, nms_bbox_scores = self.graph_builder([bbox, bbox_scores]); # graph.shape = (n, n)
    groups = self.subgraph(graph); # generate connected components
    text_lines = list();
    for index, indices in enumerate(groups):
      text_line_boxes = tf.gather(nms_bbox, indices); # text_line_boxes.shape = (m, 4)
      xmin = tf.math.reduce_min(text_line_boxes[...,0]); # xmin.shape = ()
      xmax = tf.math.reduce_max(text_line_boxes[...,2]); # xmax.shape = ()
      half_width = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) * 0.5; # half_width.shape = ()
      # fit curve with upper left corner coordinates (ul_x, ul_y)
      ul_y, ur_y = self.fit_y(text_line_boxes[...,0], text_line_boxes[...,1], xmin + half_width, xmax - half_width);
      # fit curve with down left corner coordinates (ul_x, dr_y)
      dl_y, dr_y = self.fit_y(text_line_boxes[...,0], text_line_boxes[...,3], xmin + half_width, xmax - half_width);
      # get text line score by averaging box weights
      score = tf.math.reduce_mean(tf.gather(nms_bbox_scores, indices)); # score.shape = (m, 1)
      text_lines.append((xmin, min(ul_y, ur_y), xmax, max(dl_y, dr_y), score));
    return text_lines;

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
