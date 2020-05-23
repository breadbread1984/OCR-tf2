#!/usr/bin/python3

import cv2;
import tensorflow as tf;
from create_dataset import ocr_parse_function, SampleGenerator;
from TextRecognizer import TextRecognizer;

def main():

  generator = SampleGenerator(10);
  text_recognizer = TextRecognizer();
  testset = tf.data.Dataset.from_generator(generator.gen, (tf.float32, tf.int64), (tf.TensorShape([32, None, 3]), tf.TensorShape([None,]))).repeat(-1).map(ocr_parse_function).batch(1).prefetch(tf.data.experimental.AUTOTUNE);
    # restore from existing checkpoint
  for image, label in testset:
    text = text_recognizer.recognize(image, preprocess = False);
    print(text);
    cv2.imshow("image", ((image[0,...] / 2 + 0.5) * 255.).numpy().astype('uint8'));
    cv2.waitKey();

if __name__ == "__main__":

  assert True == tf.executing_eagerly();
  main();

