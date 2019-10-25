---
layout: post
title:  "深度学习-keras入门"
date:   2019-10-19 18:44:42 +0800
categories: jekyll update
---
# 1 基本模型

## 1.1 模型堆叠
模型方法,用于配置训练模型：
{% highlight ruby %}
tf.keras.Sequential()
{% endhighlight %}
添加层：
{% highlight ruby %}
model.add(layers.Dense(32,activation="relu"))
{% endhighlight %}

## 1.2 层的配置
`activation`:设置该层的激活函数，这个参数的值可以是内置函数的名称或是可调用的对象，默认情况下无激活函数
{% highlight ruby %}
layers.Dense(32,activation="sigmoid")
layers.Dense(32,activation=tf.sigmoid)
{% endhighlight %}
`kernel_initializer` 和 `bias_initializer`:初始化层的权重和偏置，这个参数的值可以是名称或可调用对线，默认为'Glorot uniform'即均匀分布
{% highlight ruby %}
layers.Dense(32, kernel_initializer='orthogonal')
layers.Dense(32, kernel_initializer=tf.keras.initializers.glorot_normal)
{% endhighlight %}
`kernel_regularizer` 和 `bias_regularizer`：加在权重和偏置上的正则化方案，默认情况下不使用正则化函数
{% highlight ruby %}
layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01))
layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l1(0.01))
{% endhighlight %}
**L1正则化**对应的惩罚项为**L1范数**,每次更新会减去（为正）/加上（为负）一个常数，容易产生特征系数为0的情况，让特征变得稀疏，适用于**特征选择**;**L2正则化**对应的惩罚项为**L2范数**，每次更新会对特征系数进行一个比例的缩放，适用于防止模型**过拟合**。

# 2 模型的训练评估
## 2.1 配置模型的学习方法
{% highlight ruby %}
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
             loss=tf.keras.losses.categorical_crossentropy,
             metrics=[tf.keras.metrics.categorical_accuracy])
{% endhighlight %}
`model.compile()`用于配置训练模型，参数`optimizer`设置优化函数，`loss`设置损失函数，`metrics`设置评价函数，不会用于训练过程中。
## 2.2 训练模型数据
### 2.2.1 传入numpy数据训练
{% highlight ruby %}
import numpy as np
# 训练集
train_x = np.random.random((1000, 72))
train_y = np.random.random((1000, 10))
# 验证集
val_x = np.random.random((200, 72))
val_y = np.random.random((200, 10))
#测试集
test_x = np.random.random((1000, 72))
test_y = np.random.random((1000, 10))
# 将数据传入模型
model.fit(train_x, train_y, epochs=10, batch_size=100,
          validation_data=(val_x, val_y))
# 进行模型评估
model.evaluate(test_x, test_y, batch_size=32)
# 预测
result = model.predict(test_x, batch_size=32)
print(result)
{% endhighlight %}
将训练集和验证集传入`model.fit()`中，开始训练模型。在`model.fit()`中设置`epoch`和`batch`的大小。
`model.evaluate()`返回的是在`model.compile()`中设置的损失函数`loss`的值和评价函数`metrics`的值。  
`model.predict()`，为输入的样本生成输出预测
### 2.2.2 传入tf.data数据
{% highlight ruby %}
# 读入数据的特征和标签
dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
# 设置数据的
dataset = dataset.batch(32)
# 将数据复制n次，视为一个数据集
dataset = dataset.repeat()
val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
val_dataset = val_dataset.batch(32)
val_dataset = val_dataset.repeat()

model.fit(dataset, epochs=10, steps_per_epoch=30,
          validation_data=val_dataset, validation_steps=3)
test_data = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_data = test_data.batch(32).repeat()
model.evaluate(test_data, steps=30)
# 预测
result = model.predict(test_x, batch_size=32)
print(result)
{% endhighlight %}
`steps_per_epoch x batch_size = epoch`。一次`iteration`使用`batchsize`个样本训练一次，一个`epoch`使用全部训练样本训练一次。  


# 3 高级模型
`tf.keras.Sequential`模型是层的简单堆叠，无法表示任意模型。要构建复杂的模型，可以使用Keras函数式API。
使用函数式API构建的模型，层实例可调用并返回张量。输入张量与输出张量用于定义`tf.keras.Model`实例。训练方式与`tf.keras.Sequential`模型一样。
