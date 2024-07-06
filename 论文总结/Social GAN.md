<p style="text-align: center; font-size: 2.5em; color: green;">
    <strong>Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks</strong>
</p>

***

* [code](https://github.com/agrimgupta92/sgan)
* ETH
* [2018 CVPR]([Social GAN: Socially Acceptable Trajectories With Generative Adversarial Networks (thecvf.com)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Gupta_Social_GAN_Socially_CVPR_2018_paper.pdf))

## 一、概述

针对于行人轨迹预测,挑战如下:

* interpersonal:每个人的运动轨迹都会受到周围人的影响,对这些依赖关系进行建模是一种挑战.
* Socially Acceptable:一些轨迹在物理上是可行的,但是社会上是不可接受的.行人受到路权或尊重个人空间等社会规范的约束,这些是不易进行建模的.
* Multimodal:行人轨迹是多模态的.在给定历史轨迹,预期其未来轨迹应有多种可能.

现有方法虽然能解决上述挑战,但存在两点局限性,本文针对之前的局限性提出解决方案(也是本文的亮点).

* 现有的方法能对交互进行建模,但是不够高效,故<u>本文提出里一种高效的池化策略</u>.
* 之前的方法常用L2损失(即预测轨迹与真实轨迹之间的距离)评价轨迹.不利于多模态.<u>我们提出了多样性损失,鼓励GAN生成器生成多种'社会可接受的轨迹'.</u>

## 二、模型

### 2.1 问题定义

场景中所有人物的轨迹作为输入:
$$
X = {X_1,X_2,\ldots,X_n} \tag{1}
$$
$n$表示人数,$X_i = (x_i^t,y_i^t)~~(t=1,\ldots,t_{obs})$

输出为预测的轨迹:
$$
\hat{Y} = \hat{Y_1},\ldots,\hat{Y_n} \tag{2}
$$
其中第$i$个人的预测轨迹表示为$\hat{Y_i}=(x_i^t,y_i^t)~~(i = t_{obs}+1,\ldots,t_{pred})$

### 2.2 GAN简介

GAN由两个相互对抗的神经网络(G和D)组成.其中生成器G负责训练数据分布以生成数据,判别器D用于判断数据是来源于真实数据还是G.生成器需要尽量生成与真实数据一样的分布,以骗过判别器.

模型的损失函数如下:
$$
\begin{aligned}&\min_G\max_DV(G,D)=\mathbb{E}_{x\sim p_{data}(x)}[\log D(x)]+\mathbb{E}_{z\sim p_{(z)}}[\log(1-D(G(z)))]\end{aligned} \tag{3}
$$
公式中$z$是从噪声中采样的输入,$x$是从真实数据中采样的输入.在判别器完美的情况下,$D(x)=1,D(G(z))=0$,也就是最大化公式(3).在生成器完美的情况下,$D(G(z))=1$,也就是最小化公式(3).训练过程中判别器与生成器不断对抗,最后达到平衡.

GAN的训练过程如下,每个训练回合先迭代k次判别器,再迭代1次生成器.

<div align = "center"><img src = "https://cdn.jsdelivr.net/gh/liu-tian-peng/blog-img@img/img/202401281400682.png" width=100%></div><br>  
<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">Figure1. GAN训练过程伪代码</center> 

### 2.3 模型结构

模型由生成器G、判别器D和池化模块PM三个部分组成.生成器为基于LSTM的Encoder-Decoder架构.

<div align = "center"><img src = "https://cdn.jsdelivr.net/gh/liu-tian-peng/blog-img@img/img/202401281404806.png" width=100%></div><br>  
<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">Figure2. social GAN模型结构</center> 

#### 2.3.1 Generator

* Encoder

首先将相对轨迹通过MLP进行嵌入，然后再输入LSTM对历史轨迹进行编码.
$$
\begin{aligned}
e_i^t& =\phi(x_i^t,y_i^t;W_{ee})  \\
h_{ei}^{t}& =LSTM(h_{ei}^{t-1},e_i^t;W_{encoder}) 
\end{aligned} \tag{3}
$$
其中$x_i^t,y_i^t$是相对于后一时刻的位置变化量(relative).$\phi$是单层MLP.历史轨迹(relative)的嵌入输入LSTM得到最后时刻的隐藏状态,即途中斜线矩形.之后将$t_{obs}$时刻输出的隐藏状态输入池化模块PM.

* PM

池化模块以所有行人历史轨迹的编码和$t_{obs}$时刻的绝对位置作为输入.

<div align = "center"><img src = "https://cdn.jsdelivr.net/gh/liu-tian-peng/blog-img@img/img/202401282012323.png" width=100%></div><br>  
<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">Figure3. 池化策略</center> 

以计算上图中Person 1的池化向量$P_1$为例.

* 首先计算另外两人相对于Person 1的$t_{obs}$时刻的坐标,将计算的坐标通过一个带有Relu和BN层的多层MLP得到坐标的嵌入向量(Figure 3中实心矩形).
* 将Figure 2中计算得到的历史轨迹(斜线矩形)的编码与当前坐标的嵌入向量(实心矩形)拼接.
* 将拼接后的向量通过一个多层MLP,将MLP的输出向量在neighbor维度上取最大值,最终得到一个池化向量$P_1$.

每个行人最终都会得到一个包括所有neighbor的池化向量.

* Decoder

  传统的GAN将噪声作为输入并生成样本。<u>本文为了生成与过去一致的历史轨迹，通过初始化隐藏状态调节轨迹的生成.</u>隐藏状态的初始化过程如下:
  $$
  \begin{aligned}c_i^t&=\gamma(P_i,h_{ei}^t;W_c)\\h_{di}^t&=[c_i^t,z]\end{aligned} \tag{4}
  $$
  下标$i$表示行人的索引.$P_i$表示第$i$个人的池化向量,$h_{ei}^t$是对历史轨迹的LSTM编码(公式3),$W_c$是嵌入权重,$\gamma$是带有ReLU的MLP层.然后将嵌入好的向量$c_i^t$与噪声$z$拼接得到Decoder的初始化权重$h_{di}^t$.
  
  <u>先前的工作都是预测轨迹的高斯分布,然后再从预测的高斯分布中采样得到轨迹,采样的过程是不可微的,不利于反向传播.故Decoder直接预测坐标而不是高斯分布.</u>
  
  初始化Decoder隐藏状态后,其解码过程如下:
  $$
  \begin{aligned}
  e_i^t& =\phi(x_i^{t-1},y_i^{t-1};W_{ed})  \\
  P_{i}& =PM(h_{d1}^{t-1},...,h_{dn}^t)  \\
  h_{di}^{t}& =LSTM(\gamma(P_{i},h_{di}^{t-1}),e_{i}^{t};W_{decoder})  \\
  (\hat{x}_i^t,\hat{y}_i^t)& =\gamma(h_{di}^{t}) 
  \end{aligned} \tag{5}
  $$
  
  
  $e_i^t$是最新位置变化的编码值,将编码值$e_i^t$和上一时刻的隐藏状态$h_{di}^{t-1}$以及$P_i$输入LSTM更新隐藏状态.更新后的隐藏状态直接输入MLP映射到坐标维度,得到当前时间步的预测结果.利用预测结果更新当前坐标,并更新$P_i$(这样预测的每一步所有人的轨迹都可以进行交互),又进行新一轮的预测.$\gamma(P_{i},h_{di}^{t-1})$用于初始化隐藏状态,使每步的隐藏状态输入既有行人的历史轨迹信息,又有包含邻居的信息.

#### 2.3.2 Discriminator

判别器就是一个简单的LSTM编码器.以真实轨迹$T_{real}=[X_i,Y_i]$或者预测的轨迹$T_{fake}=[X_i,\hat{Y_i}]$作为输入,其中$T_{fake}$包括一段观察到的轨迹和一段预测的轨迹.将LSTM输出的隐藏状态映射到1维,得到最后的分数.分数为1则说明是真实轨迹.



