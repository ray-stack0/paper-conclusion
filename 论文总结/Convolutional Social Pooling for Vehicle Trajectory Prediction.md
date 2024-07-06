**Convolutional Social Pooling for Vehicle Trajectory Prediction**

***

[TOC]

* [code](https://github.com/nachiket92/conv-social-pooling)
* [NGSIM](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj)
* [2018 CVPR](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w29/Deo_Convolutional_Social_Pooling_CVPR_2018_paper.pdf)

***

## 一、概述



## 1.1 摘要

提出了LSTM encoder-decoder架构，该模型以 `convolutional social pooling`作为`social pooling layer`的改进，可以更稳健的学习车辆运动之间的相互依赖性，此外根据机动类别输出多模态的预测轨迹分布。最后使用NGSIM US-101、I-80评估该模型。

## 1.2 相关工作

* Convolutional Social Pooling:提出了 convolutional social pooling可以代替social LSTM中的social pooling layer.采用卷积层和最大池化层处理social tensor(编码器输出)，而不是全连接层。***采用全连接层会破坏social tensor之间的空间结构，比如空间上相邻较近的两个social tensor与相邻较远的两个social tensor在全连接层中效果是一样的，不会因为空间位置的不同而不同.这将导致泛化问题.***
* Maneuver based decoder:生成6(2*3)个机动类别的概率分布，并预测每种机动类别的可能性，从而解释车辆运动的多模态性质。

我们的模型可以归类为`maneuver based models and interaction aware models `.

* maneuver based models:大致可以分为机动类别识别模块和轨迹预测模块.前者通过历史轨迹、状态以及周围环境对车辆运动类别进行分类，后者基于运动类别输出相应的预测轨迹。
* interaction aware models:该类模型考虑车辆交互对运动的影响.

## 1.3 问题建模

* 参考系

  采用固定参考系,原点固定在$t$时刻预测的车辆上,$y$为车道前进方向,$x$为垂直于高速公路的方向.如下图

  <div align = "center"><img src = "https://cdn.jsdelivr.net/gh/liu-tian-peng/blog-img@img/img/202401111226707.png" width=100%></div><br>  
  <center style="font-size:14px;color:#C0C0C0;text-decoration:underline">Figure1. 坐标系与机动类别</center> 

  

* 输入与输出

  输入为被预测车辆所在车道及左右两车道的所有车辆(y方向±90ft的所有车辆，如上图)的历史轨迹,输入如下
  $$
  X=[X^{t-t_h},\ldots,X^{t}] \tag{1}
  $$
  其中$X^t=[x^t_0,y^t_0,\ldots,x^t_n,y^t_n]$,表示在$t$时刻$n$个车辆的坐标信息。
  
  模型输出表达为：
  $$
  Y = [Y^{t+1},\ldots,Y^{t+t_f}] \tag{2}
  $$
  其中$Y^t=[x^t_0,y^t_0]$,表示被预测车辆未来的坐标.

* Maneuver classes(机动类别)

  机动类别分为横向、纵向,横向包括向左变道、向右变道和保持车道,纵向包括制动和正常行驶,如Figure1.如果预测时间段内的平均速度小于已知时间段速度的0.8倍,则认为是在执行刹车.

## 二、模型结构

<div align = "center"><img src = "https://cdn.jsdelivr.net/gh/liu-tian-peng/blog-img@img/img/202401141129873.png" width=100%></div><br>  
<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">Figure2. 模型结构</center> 

## 2.1 Encoder

使用普通LSTM分别对每个车辆进行编码。输入是车辆坐标的序列，通过全连接层对2维坐标进行嵌入，将嵌入的坐标序列输入LSTM进行编码，取LSTM最后时刻的隐藏状态作为输出。

## 2.2 Convolutional Social Pooling

该部分以Encoder的输出作为输入.分别处理预测车辆和邻居车辆的编码。

* 邻居车辆:根据邻居车辆相对于被预测车辆的位置,将邻居车辆的编码放入13*3的网格中,没有车辆的网格用0填充,由此形成social tensor.然后再依次通过kernel size为$3 \times 3、3\times 1$的卷积和$2 \times1$的最大池化得到social context.
* 被预测车辆:通过全连接层将被预测车辆的编码映射到32维得到vehicle dynamics.

最后将social context与vehicle dynamics在最后一维进行拼接得到该部分的输出.

## 2.3 Decoder

该部分主要对被预测车辆机动类别进行分类以及预测车辆的轨迹。

* 机动类别预测:分别通过两个全连接层将convolutional social pooling的输出映射到二维和三维,经过softmax即可得到最后横向、纵向机动类别的预测结果.
* 轨迹预测:轨迹预测在训练和测试时略有不同.
  * train:直接将convolutional social pooling的输出与被预测车辆实际的机动类别(one hot encoding,横向$1\times3$,纵向$1\times2$)在最后一维拼接,再复制25份作为序列长度(被预测序列)输入LSTM,最后输出序列长度为25的隐藏状态,将隐藏状态的维度映射到5维,即可得到每一个时间步的二维高斯分布参数.
  * test:由于横向3种,纵向2种,两者线性组合共有6种机动类别.分别将6种机动类别的one-hot encoding与convolutional social pooling的输出进行拼接,即可得到被预测车辆在每一种机动类别下的预测轨迹.

## 三、关键代码

* model.py 

  <div align = "center"><img src = "https://cdn.jsdelivr.net/gh/liu-tian-peng/blog-img@img/img/202401112237455.png" width=100%></div><br>  
  <center style="font-size:14px;color:#C0C0C0;text-decoration:underline">Figure3. 模型forward方法输入参数</center> 

  ***注意输入序列长度为16，是因为作者为了适当简化模型,将NGSIM数据集的30帧历史轨迹和50帧预测轨迹经过2倍下采样.***最终得到历史轨迹序列长度为16,预测序列长度为25.

  <div align = "center"><img src = "https://cdn.jsdelivr.net/gh/liu-tian-peng/blog-img@img/img/202401112249786.png" width=100%></div><br>  
  <center style="font-size:14px;color:#C0C0C0;text-decoration:underline">Figure4. Encoder代码</center> 

  注意`hist_enc`表示被预测车辆的编码,`nbrs_enc`表示被预测车辆所有邻居的编码.

  <div align = "center"><img src = "https://cdn.jsdelivr.net/gh/liu-tian-peng/blog-img@img/img/202401112251453.png" width=100%></div><br>  
  <center style="font-size:14px;color:#C0C0C0;text-decoration:underline">Figure6. Convolutional Social Pooling代码</center> 

  `soc_enc`即为Figure2中的social tensor,只不过代码中多了一个batch维度.masks为掩码,用于指示哪些grid上有车辆,通过masked_scatter_方法依次将nbrs_enc根据masks的值复制到soc_enc对应的维度.

  <div align = "center"><img src = "https://cdn.jsdelivr.net/gh/liu-tian-peng/blog-img@img/img/202401112253928.png" width=100%></div><br>  
  <center style="font-size:14px;color:#C0C0C0;text-decoration:underline">Figure5. Decoder代码</center> 

  `self.decode`用于解码得到高斯分布的参数,其具体代码如下:

  ``` python
      def decode(self,enc):
          # enc.shape [N,117] -> [25,N,117]
          enc = enc.repeat(self.out_length, 1, 1)
          # [25,N,117] -> [25,N,128]
          h_dec, _ = self.dec_lstm(enc)
          # [25,N,128] -> [N,25,128]
          h_dec = h_dec.permute(1, 0, 2)
          # [N,25,128] -> [N,25,5] 分别预测25个时间步的高斯分布参数
          fut_pred = self.op(h_dec)
          # [N,25,5] -> [25,N,5]
          fut_pred = fut_pred.permute(1, 0, 2)
          # 为sigma_x、sigma_y、rho(相关系数)添加激活函数
          fut_pred = outputActivation(fut_pred)
          return fut_pred
  ```

  ***forward函数最后返回fut_pred(轨迹预测结果,二维高斯分布的参数)、lat_pred(横向机动类别预测结果)、lon_pred(纵向机动类别预测结果).***

  
