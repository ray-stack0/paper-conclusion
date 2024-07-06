<p style="text-align: center; font-size: 2.5em; color: green;">
    <strong>Maneuver-Aware Pooling for Vehicle Trajectory Prediction</strong>
</p>

* [code](https://github.com/m-hasan-n/pooling)
* [NGSIM](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj)
* [arXiv2104](https://arxiv.org/abs/2104.14079)

本文与[2018 CVPR Convolutional Social Pooling for Vehicle Trajectory Prediction](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w29/Deo_Convolutional_Social_Pooling_CVPR_2018_paper.pdf)[^1]的工作十分相似.所以本文只讲解两篇文章的不同之处.本文为了提高车辆执行换道操作和匝道汇入主路时的预测准确性,引入了一种基于车辆径向速度与位置的池化策略.不同之处有两处:

* 采用的坐标系与输入维度不同.
* 池化策略不同.

### 一、模型输入

由于采用不同的池化策略,故模型输入也略有改变.
$$
X = [x^{t-t_h},\ldots,x^{t-1},x^t] \tag{1}
$$
其中$t_h$表示历史轨迹的长度,$t$时刻表示当前时刻.$x^k$表示$k$时刻所有的历史轨迹输入,表示如下:
$$
x^k=[r_0^k,\phi_0^k,V_{r_0}^k,\ldots,r_n^k,\phi_n,V_{r_n}^k] \tag{2}
$$
下标表示车辆索引,其中0表示ego car,$1 \sim n$表示neighbor car.$r,\phi$是极坐标参数,$V_r$是径向速度,详细公式如下:
$$
r_i^j = \sqrt{(x_i^j-x_0^t)^2+(y_i^j-y_0^t)^2}\\
\phi_i^j=arctan(\frac{y_i^j-y_0^t}{x_i^j-x_0^t})	\tag{3}
$$
根据公式(3)可以看出,该极坐标以t时刻ego car所在位置为原点,$x$轴正方向为极轴.

$j$时刻车辆$i$相对于$t$时刻ego car的径向速度表示为$V_{r_i}^j$,具体公式如下
$$
V_{r_i}^j=V_i^jcos(\theta_i^j-\phi_i^j) \tag{4}
$$
$V_i^j $(标量差)是$j$时刻车辆$i$相对于极坐标原点的速度(也就是t时刻ego car的速度).$\theta_i^j$是$j$时刻车辆$i$的朝向,$\phi_i^j$同公式(3).

<div align = "center"><img src = "https://cdn.jsdelivr.net/gh/liu-tian-peng/blog-img@img/img/202401202257187.png" width=80%></div><br>  
<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">Figure1. 公式(4)示意图</center> 



### 二、池化策略

本文并没有类似于参考文献[^1]那样形成一个[13,3]的social tensor，然后使用卷积和最大池化对social tensor进行特征提取得到所有邻居车辆的最终编码.本文直接将所有邻居车辆的编码在车辆这一维度cat,然后再编码维度使用最大池化.

<div align = "center"><img src = "https://cdn.jsdelivr.net/gh/liu-tian-peng/blog-img@img/img/202401211246122.png" width=80%></div><br>  
<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">Figure2. 池化策略</center> 

本文对于邻居车辆的定义与文献[^1]一样，如下图.

<div align = "center"><img src = "https://cdn.jsdelivr.net/gh/liu-tian-peng/blog-img@img/img/202401211825195.png" width=80%></div><br>  
<center style="font-size:14px;color:#C0C0C0;text-decoration:underline">Figure3. neighbor car grid</center> 

[^1]:N. Deo and M. M. Trivedi, “Convolutional social pooling for vehicle trajectory prediction,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2018.