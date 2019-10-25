---
layout: post
title:  "优化函数"
date:   2019-10-20 18:44:42 +0800
categories: jekyll update
---
<html>
<head>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>
<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_CHTML">
</script>
</head>
</html>

# 1 问题定义
## 1.1 符号
$w$：待优化参数，$f(w)$：目标函数，$\alpha$：初始学习率，$g_t=\nabla f(w)$：梯度。

# 2.优化方法
## 2.1 BGD 批量梯度下降
沿着负梯度的方向更新$w$的值，当$f(w)$是凸函数时，能求得最优解。  
缺点：在每一步都需要计算梯度，每更新一个参数都要遍历完整的训练集，耗时间、耗内存，不支持在线更新模型.  
迭代：$w_{t+1}=w_t-\alpha g_t$

## 2.2 SGD 随机梯度下降法
随机选取训练集中的一个样本来计算梯度以更新参数，速度快  
缺点：计算的梯度有一定误差，优化方向不一定是全局最优，但最终结果在最优解附近，方差较大。训练过程中需要逐渐减小学习速率。  
迭代：从训练集中选取容量为m的样本，计算梯度并更新$w_{t+1}=w_t-\alpha_t g_t$，$\alpha_t=(1-\frac t {t-1} )\alpha_0 + \frac t {t-1} \alpha_{t-1}$，其中，$\alpha_0$是初始学习率，$\alpha_{t-1}$是上一次迭代的学习率。

## 2.3 MBGD 小批量梯度下降法
在每次迭代时考虑一小部分样本，同时计算这些样本上每个参数的偏导数，对于每个优化参数，将该参数在这些样本点上的偏导数求和。  
优点：降低了SGD中方差高的问题，使收敛更稳定；可以进行矩阵优化的操作，加速训练。  
缺点：选择学习率困难；容易陷入局部最优。

## 2.4 Momentum
借助了物理中动量的概念，加入参数更新的初速度，加速收敛，即前面的梯度也参与运算。每次梯度的衰减值为$\gamma$，一般取0.9。  
优点：前后梯度方向一致时，可以加速学习；前后梯度方向不一致时，会抑制震荡。  
迭代：$v_t=\gamma v_{t-1}+\alpha g_t$，$w_{t+1}=w_t-v_t$

## 2.5 Nesterov
Momentum：根据当前点梯度+以前累加的梯度进行更新  
Nesterov：根据以前累加的梯度进行的更新，在预计点处计算梯度，结合以前累加的梯度+预计点处梯度进行更新  
迭代：$v_t=\gamma v_{t-1}+\alpha \nabla f(w_t-\gamma v_{t-1})$，$w_{t+1}=w_t-v_t$

## 2.6 Adagrad
迭代：$w_{t+1}=w_{t}- \frac \alpha {\sqrt{G_t+\epsilon}}g_t$，$G_t=G_{t-1}+g_t^2$，$\epsilon$一般取1e-8。  
优点：能够实现学习率的自动更改。随着算法的迭代，$G_t$会越来越大，整体的学习率会越来越小，因此adagrad一开始是激励收敛，慢慢变成惩罚收敛，收敛速度会越来越慢。  
缺点：随着时间推移，模型最后会变得无法学习，在深度学习中，深度过深时会造成训练提前结束。

## 2.7 Adadelta
adagrad将以前的所有偏导都累加起来，adadelta控制了累加的范围  
迭代：$$ E[g^2]_t  = \gamma E[g^2]_{t-1} + (1- \gamma ) g_t^2 $$ ,$$ w_{t+1}=w_{t}- \frac \alpha {\sqrt{E(g^2)_t+\epsilon}}g_t $$ 。$$\gamma$$一般取0.9。

## 2.8 RMSprop
当adadelta中的 $\gamma=0.5$时，可以得到$$RMS[g]_t=\sqrt{E[g^2]_t+\epsilon}$$，迭代公式为：$$ w_{t+1}=w_{t}- \frac \alpha {RMS[g]_t}g_t $$。$\gamma$取0.9，学习速率为0.001。  
特点：依旧依赖全局学习率，适合处理非平稳目标，对RNN效果好。

## 2.9 Adam
迭代：$ w_{t+1} = w_{t}- \alpha \frac {\hat{m_t}} {\sqrt{\hat{n_t}+\epsilon} } $。  
其中$$m_t=\mu m_{t-1}+(1-\mu)g_t$$，$$n_t=\nu n_{t-1}+(1-\nu)g_t^2$$，$$\hat{m_t}=\frac{m_t}{1-\mu^t}$$，$$\hat{n_t}=\frac{n_t}{1-\nu^t}$$，$\mu$取0.9，$\nu$取0.999，$\epsilon$取1e-8。   
优点：结合了adagrad善于处理系数梯度和RMSprop善于处理非平稳目标的优点；对内存需求较小；为不同的参数计算不同的自适应学习率；适用于大多非凸优化、大数据集和高维空间。
## 2.10 Nadam
adam+Nesterov，梯度的计算改为$g_t=\nabla f(wt-\gamma v_{t-1})$，其中$v_t=\gamma v_{t-1}+\alpha \nabla f(w_t-\gamma v_{t-1})$

# 3 优化方法对比
## 3.1 选择
1.如果数据是稀疏的，就用自适用方法，即 Adagrad, Adadelta, RMSprop, Adam。RMSprop, Adadelta, Adam 在很多情况下的效果是相似的。Adam 就是在 RMSprop 的基础上加了 bias-correction 和 momentum，随着梯度变的稀疏，Adam 比 RMSprop 效果会好。  
2.SGD 虽然能达到极小值，但是比其它算法用的时间长，而且可能会被困在局部最小值。如果需要更快的收敛，或者是训练更深更复杂的神经网络，需要用一种自适应的算法。

## 3.2 adam缺点
1.可能不收敛  
2.可能错过全局最优