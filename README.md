# OCR-tf2
this project implements text area detection and OCR

## download the dataset for text area detection

download the dataset prepared by the author of the paper "Detecting Text in Natural Image with Connectionist Text Proposal Network" [here](https://pan.baidu.com/s/1nbbCZwlHdgAI20_P9uw9LQ)

## create dataset for text area detection

create with the following command

```bash
python3 create_dataset.py <path/to/mlt directory>
```

## train the text area detector

train with the following command

```bash
python3 train.py
```
