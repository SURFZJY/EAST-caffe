# EAST: An Efficient and Accurate Scene Text Detector

### Introduction
This is a CAFFE re-implementation of [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155v2).

thanks to these project:

- https://github.com/argman/EAST
- https://github.com/YukangWang/TextField
- https://github.com/chuanqi305/MobileNet-SSD

The features are summarized blow:

+ OpenCV_DNN/ CAFFE inference demo.
+ Only **RBOX** part is implemented.
+ Use **MobileNet_v3** as backbone. 
+ NCNN/ MNN deploy support, Use NCNN int8 quantization, the model size can be 2M or less. Very suitable for deploy on Mobile devices. 

### Contents
1. [Installation](#installation)
2. [Download](#download)
3. [Train](#Train)
4. [Demo](#demo)
5. [Examples](#examples)

### Installation
1. Any version of caffe version > 1.0 should be ok. (suggest use the https://github.com/weiliu89/caffe/tree/ssd)
2. If the DiceCoefLoss Layer do not support, please recompile caffe with the Dice Coefficient Loss Layer (https://github.com/HolmesShuan/A-Variation-of-Dice-coefficient-Loss-Caffe-Layer) or use python version 'DiceCoefLossLayer' (The comment part in train.prototxt) as subsititution.
3. The ReLU6 layer implementation can be found in https://github.com/chuanqi305/ssd
 
### Download
1. Models trained on ICDAR 2013 (training set) 

https://pan.baidu.com/s/1_daEvvt7ur3FdXVxVKSF9A  ( extract code：krdu ) 

2. Models trained on fake_idcard with single character annotation 

https://pan.baidu.com/s/1KpG7xFPChyJMftAGR2SdYw  ( extract code：m70q )

### Train
If you want to train the model, you should provide the dataset path, in the dataset path, the images and the gt text files should be separated into two filefolders as shown as below:

```bash
train_images\   train_gts\   test_images\   test_gts\
```

and the gts content format is

```bash
x1,y1,x2,y2,x3,y3,x4,y4,recog_results
```

and run

```
python train.py --gpu 0 --initmodel my_model.caffemodel
```

If you have more than one gpu, you can pass gpu ids to gpu_list(like --gpu_list=0,1,2,3)

### Demo

Put the pretrained model into snapshot directory. 

Then run

```
python demo.py 
```

### Examples

<img src="https://github.com/SURFZJY/EAST-caffe/blob/master/results/img_123.jpg" width = "450" height = "300" alt="demo on ic13" >

<img src="https://github.com/SURFZJY/EAST-caffe/blob/master/results/a.png" width = "450" height = "300" alt="demo on idcard" >
