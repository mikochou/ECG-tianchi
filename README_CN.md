## Solutions for “合肥高新杯”心电人机智能大赛

来源：[“合肥高新杯”心电人机智能大赛](https://tianchi.aliyun.com/competition/entrance/231754/introduction "“合肥高新杯”心电人机智能大赛")

## 比赛结果
初赛 F1-score: 0.8491
复赛 F1-score: 0.9228

初赛排名(5/2353)，复赛最后b榜只成功提交了一次，所以不知道最后八个模型综合的分数。


## Models
这个问题可以看成多分类问题，我使用了ResNet 50 / 101和ResNext 50.


![](https://tva1.sinaimg.cn/large/006y8mN6gy1g7tk4gh78xj307209jmx8.jpg)


以下是各个模型的线下分数。

|#|Model Name|Score|
| ------------ | ------------ | ------------ |
|1|ResNext50|0.8923|
|2|ResNet50|0.9185|
|3|ResNet50_noweight|0.9169|
|4|ResNet101|0.9143|

我在最后的全连接加入了性别和年龄两个特征，同时我也测试了去掉这两个特征的模型，虽然单模型会下降，但是ensemble之后总体分数会升高。

|#|Model Name|Score|
| ------------ | ------------ | ------------ |
|1|ResNext50|0.8904|
|2|ResNet50|0.9199|
|3|ResNet50_noweight|0.9166|
|4|ResNet101|0.9160|

八模型ensemble后线下分数：

|#|Model Name|Score|
| ------------ | ------------ | ------------ |
|1|eight models|0.9216|

最高线下分数0.9226，由以下模型组成：

|#|Model Name|
| ------------ | ------------|
|1|ResNet50|
|2|ResNet50_noweight|
|3|ResNet101|
|4|ResNext50_basic|
|5|ResNet50_basic|
|6|ResNet50_noweight_basic|



## 环境
Hardware: NVIDIA V100 (32 GB) GPU * 1
Operating System: CentOS Linux 7
Software: Python 3.7, PyTorch 1.1 

## 代码

a. Create a conda virtual environment and activate it.

```shell
conda create -n tianchi python=3.7 -y
conda activate tianchi
```

b. Install PyTorch stable or nightly and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
pip install requirements.txt
```
c. 数据下载

```shell
cd data
./data_download.sh
```
d. 数据预处理

```shell
cd code
python data_preparing.py
```

数据增强部分我参考了这个大佬的代码：
[https://github.com/JavisPeng/ecg_pytorch?spm=5176.12282029.0.0.3d952737ec5tuc](https://github.com/JavisPeng/ecg_pytorch?spm=5176.12282029.0.0.3d952737ec5tuc)

#### Loss:
因为数据是长尾数据，我使用了weighted binary_cross_entropy，每个类的权重取决于每个类的数量。
```shell
weight = 1 / log( sum(classes)  + 1e-5)
```

#### 训练模型:
```shell
cd code
./train.sh configs/dev_testA/ResNet50.yaml
./train.sh configs/dev_testA/ResNet50_noweight.yaml
./train.sh configs/dev_testA/ResNet101.yaml
./train.sh configs/dev_testA/ResNeXt50.yaml
python configs/ensemble.py
```

如果要训练不含年龄和性别特征的模型，将模型对应的yaml文件里的model name后加上"Basic"


#### 线下分数:
```shell
python main.py --configs=configs/ensemble.yaml -v
```
#### 输出测试集结果:
```shell
python main.py --configs=configs/ensemble.yaml -e
```



If you have any questions, feel free to connact:
Yuzhe Zhou
HongKong University
Email: yuzhe36 DOT connect DOT hku DOT hk
