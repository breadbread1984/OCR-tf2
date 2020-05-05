#!/usr/bin/python3

import sys;
from os import listdir, mkdir;
from os.path import join, exists, splitext;
from math import ceil;
import numpy as np;
import cv2;
from PIL import Image, ImageDraw, ImageFont;
import tensorflow as tf;
from tokenizer import Tokenizer;

def ctpn_parse_function(serialized_example):
    
  feature = tf.io.parse_single_example(
    serialized_example,
    features = {
      'data': tf.io.FixedLenFeature((), dtype = tf.string),
      'shape': tf.io.FixedLenFeature((3,), dtype = tf.int64),
      'objects': tf.io.VarLenFeature(dtype = tf.float32),
      'obj_num': tf.io.FixedLenFeature((), dtype = tf.int64)
    }
  );
  shape = tf.cast(feature['shape'], dtype = tf.int32);
  data = tf.io.decode_jpeg(feature['data']);
  data = tf.reshape(data, shape);
  data = tf.cast(data, dtype = tf.float32);
  obj_num = tf.cast(feature['obj_num'], dtype = tf.int32);
  objects = tf.sparse.to_dense(feature['objects'], default_value = 0);
  objects = tf.reshape(objects, (obj_num, 4));
  return data, objects;

def ocr_parse_function(data, label):

  data = tf.cast(data, dtype = tf.float32) / 255.;
  label = tf.cast(label, dtype = tf.int64);
  return data, label;

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

class SampleGenerator(object):

  def __init__(self, length = 4):

    self.tokenizer = Tokenizer();
    self.bg_imgs = [cv2.imread(join('background', bg_img)) for bg_img in listdir('background')];
    self.fonts_path = [join('fonts', font_path) for font_path in listdir('fonts')];
    self.length = length;
    
  def vocab_size(self):

    return self.tokenizer.size();

  def gen(self):
    
    bg_img = self.bg_imgs[np.random.randint(low = 0, high = len(self.bg_imgs))];
    tokens = np.random.randint(low = 0, high = self.tokenizer.size(), size = (self.length));
    s = self.tokenizer.translate(tokens);
    samples = list();
    for i in range(len(tokens)):
      token = tokens[i];
      ch = s[i];
      height = 32;
      width = np.random.randint(low = height - 12, high = height - 7) if ch.isdigit() else ( 
          np.random.randint(low = height - 7, high = height - 3) if ord('A') < ord(ch) < ord('Z') or ord('a') < ord(ch) < ord('z') else 
          np.random.randint(low = height - 5, high = height + 1));
      font_size = np.random.randint(low = height - 2, high = height + 2) if ch.isdigit() else (
          np.random.randint(low = height - 4, high = height + 1) if ord('A') < ord(ch) < ord('Z') or ord('a') < ord(ch) < ord('z') else
          np.random.randint(low = width - 4, high = width + 1));
      ul_xy = (np.random.randint(low = 0, high = bg_img.shape[1] - width), np.random.randint(low = 0, high = bg_img.shape[0] - height));
      sample = bg_img[ul_xy[1]:ul_xy[1] + height, ul_xy[0]:ul_xy[0] + width];
      sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB);
      sample = Image.fromarray(sample);
      font = ImageFont.truetype(self.fonts_path[np.random.randint(low = 0, high = len(self.fonts_path))], font_size);
      drawer = ImageDraw.Draw(sample);
      drawer.text((0,0), text = ch, fill = tuple(np.random.randint(low = 0, high = 210, size = (3,))), font = font);
      sample.rotate(np.random.uniform(low = -5, high = 5), expand = 0);
      sample = np.asarray(sample);
      samples.append(sample);
    sample = np.concatenate(samples, axis = 1);
    width = 32 * self.length;
    if sample.shape[1] > width:
      sample = cv2.resize(sample, (width, 32));
    else:
      ul_xy = (np.random.randint(low = 0, high = bg_img.shape[1] - (width - sample.shape[1])), np.random.randint(low = 0, high = bg_img.shape[0] - 32));
      sample = np.concatenate([sample, bg_img[ul_xy[1]:ul_xy[1] + 32, ul_xy[0]:ul_xy[0] + width - sample.shape[1]]], axis = 1);
    yield sample, tokens;

if __name__ == "__main__":

  assert tf.executing_eagerly();
  if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " <dataset dir>");
    exit();
  create_dataset(sys.argv[1]);
