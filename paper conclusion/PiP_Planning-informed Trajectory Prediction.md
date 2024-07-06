<p style="text-align: center; font-size: 2.5em; color: green;">
    <strong>PiP: Planning-informed Trajectory Prediction for Autonomous Driving</strong>
</p>
[TOC]

* [code](https://github.com/Haoran-SONG/PiP-Planning-informed-Prediction)
* [NGSIM](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj),[highD](https://www.highd-dataset.com/)
* [ECCV2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660596.pdf)

## 一、概述

### 1.1 目的

人类驾驶员通常会根据自己未来的意图来对周围车辆进行预测,当驾驶员采用不同措施时,其余车辆的未来轨迹可能也会随之变化.为了实现这点，在预测时引入候选规划轨迹，根据预测的轨迹和损失函数选择最终轨迹.

### 1.2 解决方案

提出了PiP框架,该方案在预测时会==考虑自车的规划轨迹==.具体有以下两点好处:

* 预测过程中引入规划信息,可以更好的捕获车辆之间的交互，提升预测准确性。
* 为规划模块提供有价值的接口。(即对于每条候选规划轨迹生成不同的预测轨迹，模拟了ego car执行特定动作后与其他车辆的交互过程)

传统车辆规划器与PiP框架的比较:

* traditional pipeline:传统规划器的工作流程 1.计算得到几条候选轨迹. 2.使用自定义的函数对候选轨迹进行评分,其中还会考虑其它车辆的未来轨迹. 3.选择最佳的轨迹执行.
* PiP: 会针对不同的规划轨迹对其余车辆做出不同的预测,从而导致候选规划轨迹的分数发生变化,最终影响规划轨迹的选择.体现了预测与规划轨迹之间的交互.

### 1.3 主要贡献

本文主要提出了两个模块:

* Planning-coupled module:根据历史轨迹和未来规划对车辆之间的交互进行建模，减轻了驾驶行为多模态的不确定性，从而提升预测准确性。
* Target fusion module:进一步捕获车辆之间的相互依赖性(比如对target car之间进行建模)，利用卷积对车辆的未来轨迹与自车规划的依赖性进行建模

### 1.4 问题定义

本文根据ego car的规划轨迹和周围所有车辆的历史轨迹对target car进行轨迹预测

* ego car:即可以获得规划轨迹的车辆.
* target car:ego car周围需要进行轨迹预测的车辆.
* other neighbor car:ego car视线范围内不需要进行轨迹预测的车辆.

<div align = "center"><img src = "https://cdn.jsdelivr.net/gh/liu-tian-peng/blog-img@img/img/202403072110768.png" width=100%></div><br>  
<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">Figure1. 车辆类型的定义</center>  

轨迹预测问题用公式表示为:
$$
\begin{aligned}
P(Y|X,I) \\
Y =\{Y_i | v_i\in V_{tar}\} \\
X = \{X_i|v_i\in V\}
\end{aligned} \tag{1}
$$
其中Y表示所有target car的预测轨迹,X表示表示该场景所有车辆的历史轨迹.$V_{tar}$是所有target car的集合,$V$是该场景所有车辆的集合.$I$表示ego car的规划轨迹.

车辆轨迹定义为:
$$
\begin{aligned}
X_i&=\{x_i^{t-T_{obs}+1},x_i^{t-T_{obs}+2},\ldots,x_i^t\}\\
Y_i&=\{y_i^{t+1},y_i^{t+2},\ldots,y_i^{t+T_{pred}}\}\\
I&= Y_{ego}=\{y_{ego}^{t+1},y_{ego}^{t+2},\ldots,y_{ego}^{t+T_{pred}}\}
\end{aligned}
\tag{2}
$$
其中$x_i,y_i$均表示二维平面上的坐标,$T_{obs},T_{pred}$分别表示历史轨迹和预测轨迹的帧数.

## 二、模型结构

<div align = "center"><img src = "https://cdn.jsdelivr.net/gh/liu-tian-peng/blog-img@img/img/202403072140322.png" width=100%></div><br>  
<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">Figure2. 模型结构</center>  

模型结构主要分为三部分,Planning Coupled Module(对规划轨迹与历史轨迹进行建模),Target Fusion Module(对Target car之间进行建模)和Maneuver Based Decoder(对Target car的编码进行解码得到高斯分布参数).

### 2.1 输入的处理

对数据集进行处理，得到以下数据形式：

* traj:数据总数 =(data number)*(13+grid num)

  | 0          | 1          | 2        | 3       | 4       | 5       | 6                | 7                     | 8      | 9     | 10          | 11       | 12          | 13~end*                           |
  | ---------- | ---------- | -------- | ------- | ------- | ------- | ---------------- | --------------------- | ------ | ----- | ----------- | -------- | ----------- | --------------------------------- |
  | Dataset Id | Vehicle Id | Frame Id | Local X | Local Y | Lane Id | Lateral maneuver | Longitudinal maneuver | Length | Width | Class label | Velocity | Accerlation | Neighbor car Ids at grid location |

  

* tracks:形状为[Dataset Id num, Vehicle Id num],即以数据集Id和车辆Id作为索引,其中每一个元素都是该车的历史轨迹,数据量为(11+grid)*totalFramNum

  | 0        | 1       | 2       | 3       | 4                | 5                     | 6      | 7     | 8           | 9        | 10          | 11~end*                           |
  | -------- | ------- | ------- | ------- | ---------------- | --------------------- | ------ | ----- | ----------- | -------- | ----------- | --------------------------------- |
  | Frame Id | Local X | Local Y | Lane Id | Lateral maneuver | Longitudinal maneuver | Length | Width | Class label | Velocity | Accerlation | Neighbor car Ids at grid location |

将上述数据导入Python，组成batch,batch的数据格式如下(详见data.py):

| 参数名称           | 参数含义                                                     | shape                                              |
| ------------------ | ------------------------------------------------------------ | -------------------------------------------------- |
| nbsHist_batch      | 一个batch中所有邻居车辆(即target car的邻居车辆)相对于t时刻target car的历史轨迹 | [hist_len,nbs_batch_size,2]                        |
| targsHist_batch    | 所有target car相对于自身t位置的的历史轨迹                    | [hist_len,targs_batch_size,2]                      |
| targsFut_batch     | 所有target car的真实未来轨迹(绝对坐标)                       | [fut_len,targs_batch_size,2]                       |
| lat_enc_batch      | 所有target car横向动作的one-hot向量                          | [targs_batch_size,3]                               |
| lon_enc_batch      | 所有target car纵向动作的one-hot向量                          | [targs_batch_size,2]                               |
| planFut_batch      | ego car相对于所有target car t时刻坐标的规划轨迹              | [plan_len,targs_batch_size,2]                      |
| nbsMask_batch      | Observation Tensor的Mask矩阵(Fig.2),用于指示以target car为中心的哪些位置存在车辆 | [targs_batch_size, 5, 25, enc_size] 5,25是网格数量 |
| planMask_batch     | Planning Tensor的Mask矩阵,用于指示以target car为中心ego car的位置 | [targs_batch_size, 5, 25, enc_size]                |
| targsEncMask_batch | Target Tensor与Fused Target Tensor的Mask矩阵,用于指示以ego car为中心的区域内target car的空间位置 | [batch_size, 5, 25, enc_size]                      |
| targsFutMask_batch | Target car未来轨迹的Mask矩阵,为1说明存在真实轨迹.            | [fut_len,targs_batch_size,2]                       |

### 2.2 Planning Coupled Module

该模块主要对历史轨迹与规划轨迹的依赖性进行建模.该部分分为三个分支,即Planning Tensor,Observation Tensor和Dynamic Encoding三部分,最后依次将三者拼接.

1. Planning Tensor与Planning Tensor

   首先利用1D卷积和LSTM提取规划轨迹的时序信息,以LSTM最后时刻输出的隐藏状态作为轨迹的表达.shape变化为

   $[plan\_len,targs\_batch\_size,2]\rightarrow[targs\_batch\_size,temporal\_embedding\_size,plan\_len]\rightarrow[targs\_batch_size,encoder\_size]$

   然后使用masked_scatter_方法和PlanMask得到Planning Tensor.对Palnning Tensor使用$Conv2d\rightarrow MaxPool2d\rightarrow Conv2d$使邻居车辆之间的轨迹信息发生交互.==Observation Tensor形成方法类似,只是提取的是邻居车辆的历史轨迹和使用的是nbsMask矩阵.但是由于规划轨迹和历史轨迹属于不同时间域,故两者采用不同的LSTM==最终得到两个shape为[targs_batch_size,soc_conv2_depth,9,1]的Tensor.

2. Dynamic Encoding

   对所有target car进行动力学编码.首先依旧使用1D卷积和LSTM提取target car历史轨迹的时序信息,再使用全连接层将时序信息一定维度.shape变化为:

   $[hist\_len,targs\_batch\_size, 2]\rightarrow [targs\_batch\_size,dynamics\_encoding\_size]$

3. cat

   首先将Observation Tensor和Planning Tensor经过卷积和池化得到的结果进行拼接,得到shape为[targs_batch_size,soc_conv2_depth,9,2]的Tensor,再经过池化得到 [targs_batch_size,soc_conv2_depth,5,1],将其展平得到 [targs_batch_size,80],将得到的结果与Dynamic Encoding拼接得到[targs_batch_size,80+32].

   

### 2.3 Target Fusion Module

该模块主要对Target car之间的交互进行进一步建模.

首先利用上一个模块得到的每个target car的编码结果[targs_batch_size,80+32]结合targsEncMask_batch形成target tensor[batch_size, 5, 25, enc_size] ,该空间位置以ego car为中心.然后通过卷积,逆卷积,捷径分支等操作得到Fused Target Tensor.shape的具体变化见Fig2.最后得到Fused Target Tensor的shape为[batch_size, 5, 25, fuse_enc_size] .

将Target Tensor与Fused Target Tensor拼接得到[batch_size, 5, 25, enc_size+fuse_enc_size] ,再提取所有Target car的编码得到该模块的输出结果[targs_batch_size,targ_enc_size+fuse_enc_size].

### 2.4 Maneuver Based Decoder



对Target car的编码进行解码,得到二维高斯分布的5个参数.解码之前先将target car的编码与对应的横/纵向行为的one-hot向量进行拼接(训练时只拼接真实的机动行为,测试时依次拼接所有机动行为).拼接在时间维度上复制需要预测轨迹点的个数那么多份,得到[out_length,targs_batch_size,229].将拼接结果使用LSTM解码,再使用全连接层映射到5维(因为二维高斯参数个数为5),得到最终的预测结果[out_length,targs_batch_size,5] .

