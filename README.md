# Online Attention Accumulation
This work is finished by [Peng-Tao Jiang](pengtaojiang.github.io), [Qibin Hou](http://mmcheng.net/qbhou/), [Yang Cao](https://mmcheng.net/ycao), [Ming-Ming Cheng](https://mmcheng.net/cmm/), [Yunchao Wei](https://weiyc.github.io/), [Hongkai Xiong](http://min.sjtu.edu.cn/xhk.htm).
This repository contains the original code and the links for data and pretrained models. If you have any questions about our paper "Integral Object Mining via Online Attention Accumulation", please feel free to contact me (pt.jiang AT mail DOT nankai.edu.cn).

### Table of Contents
1. [Citation](#citation)
2. [Installation](#installation)
3. [Results](#results)
4. [Pytorch re-implementations](#pytorch-re-implementations)
### Citation
If you use these codes and models in your research, please cite:

### Installation
#### 1. Dependence
  ubuntu 16.04  
  python 2.7  
  [caffe dependence](https://caffe.berkeleyvision.org/install_apt.html)

#### 2. Compilie caffe
```
git clone https://github.com/PengtaoJiang/OAA.git
cd OAA/
make all -j4 && make pycaffe
```
#### 3. Download VOC 2012 dataset
```

```
#### 3. training for accumulating attention
```
cd examples/oaa/
./train.sh exp1 0
```
