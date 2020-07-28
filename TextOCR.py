#!/usr/bin/python3 

import sys;
import cv2;
import tensorflow as tf;
from TextDetector import TextDetector;
from TextRecognizer import TextRecognizer;

class TextOCR(object):

  def __init__(self):

    self.detector = TextDetector();
    self.recognizer = TextRecognizer();

  def scan(self, img):

    textlines, bbox, scores = self.detector.detect(img);
    results = list();
    for textline in textlines:
      timg = img[int(textline[1]):int(textline[3]),int(textline[0]):int(textline[2]),:];
      text, _ = self.recognizer.recognize(timg);
      results.append({'image': timg, 'text': text, 'position': textline});
    return results;

if __name__ == "__main__":

  if len(sys.argv) != 2:
    print('Usage: ' + sys.argv[0] + "<image>");
    exit();
  img = cv2.imread('test.jpg');
  if img is None:
    print('failed to open image!');
    exit();
  ocr = TextOCR();
  results = ocr.scan(img);
  for result in results:
    print(result['text']);

