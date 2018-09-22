### Toxic Comment Classification

#### Competition Introduction

Build multi-labels classification models that's able to detect different types of toxicity (malicious words) of comments from Wikipedia's talk page edits, such as threats, obscentiy, insults etc.

There are two post-competition sharing posts I think very good (in Chinese):

- [Kaggle 恶意评论(toxic comment classification)分类 top 1%方案](https://zhuanlan.zhihu.com/p/34922134)
- [通过kaggle比赛学习机器学习文本分类方法](https://zhuanlan.zhihu.com/p/34899693?group_id=961190993937268736)

#### About My Solution (221/4551)

The competition was ended half a year ago (2018.03.20). My final submission is the blend of multiple models:

- Logistic Regression/Naive Bayes - Logistic Regression/FM/LGB/MLP based on Tfidf vectorization + text statistical meta features
- Neural Nets (CNN/RNN) based on pre-trained word embeddings (FastText, GloVe)

Thanks for reading.

2018.09.22