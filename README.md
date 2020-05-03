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

## text area detection results

here are some results of my model which is enclosed at model/ctpn.h5 . As you can see, the detection (blue boxes) is OK, but the graph connected box (green ones) made in post process need to be improved.

<p align="center">
  <table>
    <caption>Loss</caption>
    <tr><td><img src="pics/ctpn/loss.png" alt="train loss" width="800" /></td></tr>
  </table>
</p>
<p align="center">
  <table>
    <caption>Detection results</caption>
    <tr>
      <td><img src="pics/ctpn/result1.png" width="400" /></td>
      <td><img src="pics/ctpn/result2.png" width="400" /></td>
    </tr>
    <tr>
      <td><img src="pics/ctpn/result3.png" width="400" /></td>
      <td><img src="pics/ctpn/result4.png" width="400" /></td>
    </tr>
  </table>
</p>
