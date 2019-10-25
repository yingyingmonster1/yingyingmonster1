---
layout: post
title:  "Keras评价指标"
date:   2019-10-23 10:44:42 +0800
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

# Keras中内置的评价函数
## 二分类
binary_accuracy：二分类问题的评价指标，即准确率
## 多分类问题
categorical_accuracy：检查y_true（one-hot向量）中最大值与y_pred中最大值对应的index是否相等。适合**多分类单标签**任务，不适合多标签任务。  
sparse_categorical_accuracy：检查y_true（本身就是index）中的值与y_pred中最大值对应的index是否相等