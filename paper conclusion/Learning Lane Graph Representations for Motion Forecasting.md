# Learning Lane Graph Representations for Motion Forecasting

[TOC]

* [论文链接]([2007.13732.pdf (arxiv.org)](https://arxiv.org/pdf/2007.13732.pdf))  

* [argoverse](https://www.argoverse.org/)

* [code]([uber-research/LaneGCN: [ECCV2020 Oral\] Learning Lane Graph Representations for Motion Forecasting (github.com)](https://github.com/uber-research/LaneGCN))

  ***

  ***

***Data:2023/12/26***

## 1. 概览

### 1.1. 摘要

提出了一种新的预测模型。该模型直接根据原始地图数据构建车道图( LaneGraph )。根据GCN进行改进，进一步得到适用于轨迹预测的LaneGCN。该模型通过对4种交互类型(lane-to-lane,lane-to-actor,actor-to-actor,actor-to-lane)进行建模，然后通过网络融合四种交互。在argoverse数据集上，其结果达到领先水平。

###  1.2. 结论与贡献

* 过往方法缺点：
  1.  对地图数据rasterization process导致信息丢失。
  2. 由于地图拓扑结构复杂，2D卷积捕获特征的效率较低。
* 主要贡献
  1. 直接根据矢量地图数据构建Lane Graph，避免信息丢失。
  2. 同时提出LaneGCN，可以有效的捕获车道的拓扑结构和长距离依赖性。
  3. 将车道和actor表示为节点，同时分别采用1D CNN和LaneGCN提取actor节点与车道节点的特征信息。然后利用空间注意力和LaneGCN对四种交互进行建模。

![方法](https://cdn.jsdelivr.net/gh/liu-tian-peng/blog-img@img/img/202312261728303.png)

### 1.3. 环境搭建

1. 创建虚拟环境

   ```ba
   conda create --name lanegcn python=3.7
   conda activate lanegcn
   conda install pytorch==1.5.1 torchvision cudatoolkit=10.2 -c pytorch # pytorch=1.5.1 when the code is release
   
   # install others dependancy
   pip install scikit-image IPython tqdm ipdb
   ```

2. 安装argoverse API

   [参考链接](https://blog.csdn.net/weixin_40633696/article/details/124965566)

   注意：执行`pip install -e.`之前应该修改setup.py文件中的sklearn包，将其修改为scikit-learn。

3. 针对分布式训练

   ```bas
   pip install mpi4py
   
   # install horovod with GPU support, this may take a while
   HOROVOD_GPU_OPERATIONS=NCCL pip install horovod==0.19.4
   
   # if you have only SINGLE GPU, install for code-compatibility
   pip install horovod
   ```

4. 数据准备

   在终端中使用`bash get_data.sh`命令之前，有以下注意事项。
   
   * 由于原来的数据集的链接已经失效，需要更改数据集的链接，最新链接参考官网。

## 2.模型结构

模型总共分为4个部分。

* ActorNet：以actor的历史轨迹(2D坐标)作为输入，然后通过1D CNN提取特征。
* MapNet：以高精度地图作为输入，然后构建Lane Graph，再使用LaneGCN提取节点特征。
* FusionNet：包括4个交互模块。A2L将actor节点的交通信息融入车道节点；L2L更新车道信息；L2A将车道信息传入actor节点；A2A实现actor之间的交互。
* Header：使用融合后的actor节点特征生成多模态轨迹。

![模型结构](https://cdn.jsdelivr.net/gh/liu-tian-peng/blog-img@img/img/202312261728491.png)

### 2.1. ActorNet

actor数据由场景中所有可观察到的actor的历史轨迹数据组成。每条轨迹被表示为位移的序列${\Delta P_{-(T-1)},\ldots,\Delta P_{-1},\Delta P_{0}}$，其中$\Delta P_t$表示从时间步$t-1~to~t$的2D位移。==所有的坐标都是在BEV中定义的。==由于有的actor时间步不足T，用0填充不足的部分，然后用$1 *T$的msak向量表征是否为填充。此时一条历史轨迹可以用$3*T$的tensor表示。

然后采用1D CNN进行特征提取。以$t=0$时刻的时间特征图作为actor的特征。有3种不同尺度的1D CNN，使用FPN结构融合多尺度特征。

### 2.2. MapNet

MapNet包含两个步骤:

1. 根据高精地图创建Lane Graph。

   地图数据由车道(中心线上一系列点)及其连接性(4种连接性)表示。

   ![image-20231226121952442](https://cdn.jsdelivr.net/gh/liu-tian-peng/blog-img@img/img/202312261728945.png)

   中心线：

   * 红色：感兴趣的车道中心线

   * 橙色：predecessor

   * 蓝色：successor

   * 紫色：left

   * 绿色：right

     将车道节点定义为上图任意两个连续的灰色圆点形成的中心线段，其节点位置为两圆点的中心坐标。车道节点之间也存在4种连接类型，如上图右侧。我们将车道节点表示为$V \in R^{N \times 2}$，$N$为节点个数，2表示2维的BEV坐标。使用4个邻接矩阵$\{A_i\}_{i \in \{pre,suc,left,right\}}$表示连接信息，其中$A_{i} \in R^{(N \times N)}$，$A_{i,jk}$表示第$i$种连接类型的邻接矩阵的第$j$行，第$k$列的元素，若该元素为1，则节点$j$和节点$k$是第$i$种连接类型。

2. 对Lane Graph使用LaneGCN提取特征。

   * 捕获局部特征：对于车道节点$i$的输入特征$x_i$定义为：

   $$
   x_i = MLP_{shape}(v_i^{end}-v_i^{start})+MLP_{loc}(v_i)
   $$

   其中$v_i^{end}、v_i^{start}$表示节点$i$的结束坐标和开始坐标，$x_i$ 是输入特征矩阵$X$的第$i$行。前一个MLP考虑车道节点的形状(大小和方向)，后一个MLP考虑车道节点的位置(中心坐标)。

   * 聚合车道拓扑信息：LaneGCN
     $$
     Y = XW_0+\sum_{i \in\{pre,suc,left,right\}}A_iXW_i
     $$

   * 增加车道方向感受野，捕获长程依赖性：Dilated LaneConv
     $$
     Y = XW_0+A_{pre}^kXW_{pre,k}+A_{suc}^kXW_{suc,k}
     $$
     $A_{pre}^k$矩阵是$A_{pre}$邻接矩阵的$k$次方，$k$次方后能聚合距离为$k$的车道节点的信息。注意只在车道方向上进行上述操作，以增加车道方向的感受野。

   * 利用多尺度特征： multi-scale LaneConv 
     $$
     Y = XW_0 + \sum_{i \in \{left,right\}}A_iXW_i+\sum_{c=1}^C(A_{pre}^{k_c}XW_{pre,k_c}+A_{suc}^{k_c}XW_{suc,k_c})
     $$
     $k_c \in \{k_1,k_2,\dots,K_C\}$，$k_c$表示第$c$个膨胀尺寸，我们用$LaneGCN(k_1,k_2,\dots,K_C)$表示多尺度层。代码中$k_c \in \{1,2,4,8,16,32\}$。其LaneGCN结构如下：

     ![image-20231226132520191](https://cdn.jsdelivr.net/gh/liu-tian-peng/blog-img@img/img/202312261729581.png)

### 2.3. FusionNet

该网络主要用于融合ActorNet和MapNet输出的acotr节点、车道节点信息。该模块包括依次4个部分。

* A2L：向车道节点(包含actor的历史轨迹信息)引入实时的交通信息。

  对于A2L、L2A、A2A使用spatial atention layer进行建模，以A2L为例：
  $$
  y_i = x_iW_0+\sum_j \phi(concat(x_i,\Delta_{i,j},x_j)W_1)W_2
  $$
  $x_i$表示第$i$个actor节点；$x_j$表示车道节点$j$；$y_i$表示融合了==context lane node==信息的acotr节点；$\phi$由规划化层和ReLU组成；

  $\Delta_{i,j}=MLP(v_j-v_i)$,$v$表示节点位置；

  context lane node：指与actor节点$i$之间的$l_2$距离小于规定阈值的车道节点。

  本文阈值规定大小如下：

  | A2L  | L2A  | A2A  |
  | ---- | ---- | ---- |
  | 7    | 6    | 10   |

  A2L具有两个残差模块，输出通道为128。==A2L、L2A、A2A具有相同的结构。==

* L2L：传播交通信息以更新车道节点。

  使用LaneGCN实现L2L。与MapNet中的LaneGCN具有相同的结构。

* L2A：将融合后的车道信息和交通信息传入actor节点。

* A2A：对actor之间的交互进行建模，并输出actor的特征用于后续的预测。

### 2.4. Prediction Header

以FusionNet输出的Actor特征作为输入，用两个预测头分别输出多模态轨迹及其置信度。对于每个actor，会生成K条轨迹及其置信度。对于actor m，其轨迹输出为：
$$
O_{m,reg}=\{(P_{m,1}^k,\ldots,P_{m,T}^k)\}_{k\in [0,K-1]}
$$
$P_{m，i}^k$表示acotr m的第k条轨迹在第$i$时间步输出的BEV预测坐标

置信度输出：
$$
O_{m,cls} = (c_{m,0},\dots,c_{m,K-1})
$$
$c_{m,i}$表示actor m的轨迹$i$的置信度。对于actor m，通过$MLP(P_{m,T}^k-P_{m,0}^k)$可以得到k个distance embedding。然后concate(k embedding,node feature)经过残差模块和线性模块最后输出k个置信度分数。

### 2.5. Loss func

$$
L = L_{cls}+\alpha L_{reg}
$$

对于分类损失，使用max-margin loss：
$$
L_{cls} = \frac{1}{M(K-1)}\sum_{m=1}^M \sum _{k\neq \hat{k}}max(0,c_{m,k}+\epsilon-c_{m,\hat{k}})
$$
$\hat{k}$表示具有最小FDE(final displacement error)对应的轨迹索引；M是actor的个数，$\epsilon$是余量。

对于回归损失，使用$smoooth~l_1~loss$：
$$
L_{reg}=\frac{1}{MT}\sum_{m=1}^{M}\sum _{t=1}^{T}reg(P_{m,t}^{\hat{k}}-P_{m,t}^*)
$$
$P_{m,t}^*$是actor m在t时刻的真实BEV坐标；$reg(x)=\sum_id(x_i)$，$x_i$是$x$的第i个元素。
$$
d(x_i)=
\begin{cases}
0.5x_i^2 & {if\lVert x_i \rVert<1} \\
\lVert x_i \rVert-0.5 & \text{othrwise}
\end{cases}
$$
$\lVert x_i \rVert$表示$x_i$的$l_1$范数。

### 2.6. 详细网络结构

![image-20231226150621350](https://cdn.jsdelivr.net/gh/liu-tian-peng/blog-img@img/img/202312261729303.png)

## 3.实验

以argoverse作为数据集，该数据采样频率为10Hz，以前2s作为历史轨迹，后3s作为Ground Truth进行预测。参照ActorNet和MapNet对数据进行处理.