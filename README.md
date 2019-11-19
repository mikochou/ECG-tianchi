## Solutions for “合肥高新杯”心电人机智能大赛

[中文版readme](./README_CN.md)

source: [“合肥高新杯”心电人机智能大赛](https://tianchi.aliyun.com/competition/entrance/231754/introduction "“合肥高新杯”心电人机智能大赛")

## Result
Round 1 F1-score: 0.8491
Round 2 F1-score: 0.9228
In round 1, I rank (5/2353). But my final result failed to be submitted in round 2 due to some committee reason, so I do not know the final online score.


## Models
I treat this as 34-class classification (in round 1, it's 55-class).
I use ResNet 50 / 101 and ResNext 50.
![](https://tva1.sinaimg.cn/large/006y8mN6gy1g7tk4gh78xj307209jmx8.jpg)
This are the scores of all models on offline validation set, which rely on to give the weights in model ensembling.

|#|Model Name|Score|
| ------------ | ------------ | ------------ |
|1|ResNext50|0.8923|
|2|ResNet50|0.9185|
|3|ResNet50_noweight|0.9169|
|4|ResNet101|0.9143|

The age and gender feature have been added to above models. I also test those models which do not have the two features. Although the validation set score of models without these two features will be lower, the online score for all models after ensemble can increase.

|#|Model Name|Score|
| ------------ | ------------ | ------------ |
|1|ResNext50|0.8904|
|2|ResNet50|0.9199|
|3|ResNet50_noweight|0.9166|
|4|ResNet101|0.9160|

ensemble score:

|#|Model Name|Score|
| ------------ | ------------ | ------------ |
|1|eight models|0.9216|

Yet the highest score on validation set is 0.9226.

|#|Model Name|
| ------------ | ------------|
|1|ResNet50|
|2|ResNet50_noweight|
|3|ResNet101|
|4|ResNext50_basic|
|5|ResNet50_basic|
|6|ResNet50_noweight_basic|

(*basic means no age and gender features)


## Environment
Hardware: NVIDIA V100 (32 GB) GPU * 1
Operating System: CentOS Linux 7
Software: Python 3.7, PyTorch 1.1 

## Run the Code

a. Create a conda virtual environment and activate it.

```shell
conda create -n tianchi python=3.7 -y
conda activate tianchi
```

b. Install PyTorch stable or nightly and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
pip install requirements.txt
```
c. data download

```shell
cd data
./data_download.sh
```
d. data prepare

```shell
cd code
python data_preparing.py
```
In the data enhancement section I referenced the following code:
[https://github.com/JavisPeng/ecg_pytorch?spm=5176.12282029.0.0.3d952737ec5tuc](https://github.com/JavisPeng/ecg_pytorch?spm=5176.12282029.0.0.3d952737ec5tuc)

#### Loss:
I used weighted binary_cross_entropy. The weight depends on the amount of every class.
```shell
weight = 1 / log( sum(classes)  + 1e-5)
```

#### Training:
```shell
cd code
./train.sh configs/dev_testA/ResNet50.yaml
./train.sh configs/dev_testA/ResNet50_noweight.yaml
./train.sh configs/dev_testA/ResNet101.yaml
./train.sh configs/dev_testA/ResNeXt50.yaml
python configs/ensemble.py
```
To train models without age and gender featrue, modify model name in yaml file to 'modelname+_Basic'

#### Validating:
```shell
python main.py --configs=configs/ensemble.yaml -v
```
#### Testing:
```shell
python main.py --configs=configs/ensemble.yaml -e
```



If you have any questions, feel free to connact:
Yuzhe Zhou
HongKong University
Email: yuzhe36 DOT connect DOT hku DOT hk
