# Sagemaker_spoken_language_identification

本项目利用AWS Sagemaker 封装多种语音识别的深度学习算法

### 准备数据

按照以下的格式准备数据：

每个语言音频约10个小时：

1、音频不含噪音，背景音；

2、不同的发音者，男女比例相当，越多样性越好；

3、每个音频文件时长10~20分钟；

4、mp3格式，单声道，采样率22050；

5、语言命名规范为: '语言代码_性别_音频名称'，比如 de_f_1233444422.mp3, de 为语言代码，f表示女声，m表示男声，123344422为音频名称；

前缀'de' 'cn'等代表不同的语种

数据分训练数据和测试数据，并且可以准备一定的噪音数据（噪音数据参考 https://shishuai-share-external.s3.cn-north-1.amazonaws.com.cn/dataset/noises.tar.gz ）。

### 第一步：数据预处理

执行 1-processing 下的 processing.ipynb

### 第二步：模型训练

执行 2-training 下的 sagemaker_train.ipynb

### 第三步：模型部署

执行 3-inference 下的 inference.ipynb
