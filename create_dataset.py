#!/usr/bin/python3

import sys;
from os import listdir, mkdir;
from os.path import join, exists, splitext;
import numpy as np;
import cv2;
import tensorflow as tf;

def parse_function(serialized_example):
    
  feature = tf.io.parse_single_example(
    serialized_example,
    features = {
      'data': tf.io.FixedLenFeature((), dtype = tf.string),
      'shape': tf.io.FixedLenFeature((2,), dtype = tf.int64),
      'objects': tf.io.VarLenFeature(dtype = tf.float32),
      'obj_num': tf.io.FixedLenFeature((), dtype = tf.int64)
    }
  );
  shape = tf.cast(feature['shape'], dtype = tf.int32);
  data = tf.io.decode_jpeg(feature['data']);
  data = tf.reshape(data, shape);
  obj_num = tf.cast(feature['obj_num'], dtype = tf.int32);
  objects = tf.sparse.to_dense(feature['objects'], default_value = 0);
  objects = tf.reshape(objects, (obj_num, 4));
  return data, objects;

def create_dataset(root_dir, rpn_neg_thres = 0.3, rpn_pos_thres = 0.7):

  if not exists('datasets'): mkdir('datasets');
  writer = tf.io.TFRecordWriter(join('datasets', 'trainset.tfrecord'));
  count = 0;
  for imgname in listdir(join(root_dir, "image")):
    imgpath = join(root_dir, "image", imgname);
    img = cv2.imread(imgpath);
    if img is None:
      print("failed to open image file " + imgpath);
      continue;
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
    labelpath = join(root_dir, "label", splitext(imgname)[0] + ".txt");
    if False == exists(labelpath):
      print("failed to open label file " + labelpath);
      continue;
    f = open(labelpath, 'r');
    # process label
    targets = list();
    for line in f.readlines():
      target = np.array(line.strip().split(',')).astype('int32');
      targets.append(target);
    targets = np.array(targets, dtype = np.float32); # targets.shape = (n, 4)
    # write sample
    trainsample = tf.train.Example(features = tf.train.Features(
      feature = {
        'data': tf.train.Feature(bytes_list = tf.train.BytesList(value = [tf.io.encode_jpeg(img).numpy()])),
        'shape': tf.train.Feature(int64_list = tf.train.Int64List(value = img.shape)),
        'objects': tf.train.Feature(float_list = tf.train.FloatList(value = targets.reshape(-1))),
        'obj_num': tf.train.Feature(int64_list = tf.train.Int64List(value = [targets.shape[0],]))}));
    writer.write(trainsample.SerializeToString());
    count += 1;
  writer.close();
  print("written " + str(count) + " samples");
  
if __name__ == "__main__":

  assert tf.executing_eagerly();
  if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " <dataset dir>");
    exit();
  create_dataset(sys.argv[1]);
