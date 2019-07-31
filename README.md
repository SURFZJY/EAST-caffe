## EAST-caffe

caffe复现EAST算法，mbv2的底层网络结构，改进unet融合模式

1. 环境要求

- python2.7/ python3+
- caffe (with python API)
  - boost 1.69
  - opencv 3.5



2. 训练数据准备

ICDAR2015数据格式，按照img和gt分成对应名称为xxx和 xxx的文件夹

```bash
train_images\   train_gts\   test_images\   test_gts\
```

其中gts\中的txt文件格式如下：

```bash
x1,y1,x2,y2,x3,y3,x4,y4,recog_results
```

3. Demo演示

4. 训练自己的数据

   ```bash
   python train.py
   ```

5. 

6. Demo演示


