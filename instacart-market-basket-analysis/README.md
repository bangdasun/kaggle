### Instacart Market Basket Analysis

#### Competition Introduction

Build algorithms that could predict whether the user will re-purchase items from its historical item list.

There are several public kernels and post-game sharing code I refer to:

- [LB 0.3805009, Python Edition](https://www.kaggle.com/nickycan/lb-0-3805009-python-edition)
- [Word2Vec for products analysis + 0.01 LB](https://www.kaggle.com/omarito/word2vec-for-products-analysis-0-01-lb)
- [Faron script applied to Sh1ng's solution](https://www.kaggle.com/cpmpml/faron-script-applied-to-sh1ng-s-solution)
- [plantsgo/Instacart-Market-Basket-Analysis](https://github.com/plantsgo/Instacart-Market-Basket-Analysis)
- [KazukiOnodera/Instacart](https://github.com/KazukiOnodera/Instacart)

#### About My Solution 

The competition was ended one year ago. I'm very interested in this recommendation-like problem, therefore even its ended I still go through the data and spend one week on it. The final submission is the "recommended list" for each order, but indeed we can expand historical item list of each user and convert it to a binary classification problem.

- User features: users profiles;
- Product features: products properties;
- User Product interaction features: user preferences of products;
- LightGBM
- Expectation Maximization of F-1 score.


I'll post the top solutions reading on zhihu shortly.

Thanks for reading.

2018.10.02