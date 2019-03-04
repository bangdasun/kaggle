### Quora Insincere Questions Classification

#### Competition Introduction

See my zhihu article: [Kaggle竞赛-Quora Insincere Question Classification小结](https://zhuanlan.zhihu.com/p/58203555)

#### About Our Solution (76/4037)

This is my first competition in team (5 people), my teammates are all awesome experts, they are (sorted alphabetically):

- [Hikkiiiiiiiii](https://www.kaggle.com/wochidadonggua)
- [huiqin](https://www.kaggle.com/qinhui1999)
- [THLUO](https://www.kaggle.com/ilovearsenal)
- [uuulearn](https://www.kaggle.com/uuulearn)

It was really a happy time to work with them, I learned a lot from them. 

This is a kernel competition, which reminds me of Mercari Price Suggestion Challenge one year ago :( But during the past year I got some experiences on kaggle, therefore this time I did better jobs in programming, development and documentation. I learned more about applying deep neural networks on text classification problem, especially the hyper-parameter tuning. 

Our final submissions were combinations of RNN and CNN, the higher one was also the one had highest local CV score. We finally got a top 2% rank which was not too bad. For now I only share the models before I merge with my teammates, main structure and parameters:

- `Embeddings(0.5 GloVe + 0.5 Paragram) -> Dropout2d(0.1) -> BiLSTM(96) -> BiGRU(96) -> Atten Layer -> MaxPool + AvgPool -> Concat -> Dense(64) -> Dropout(0.1) -> Dense(1)`, with 5-folds cross validation, `epochs=4`, `batch_size=256`, `max_len=70`, `max_features=200000`, optimizer is `Adam(1e-3)`
- `Embeddings(0.5 GloVe + 0.5 Paragram) -> Dropout2d(0.1) -> BiLSTM(128) -> BiGRU(128) -> Atten Layer -> MaxPool + AvgPool -> Concat -> Dense(96) -> Dropout(0.1) -> Dense(1)`, with 5-folds cross validation, `epochs=4`, `batch_size=512`, `max_len=70`, `max_features=200000`, optimizer is `Adam()` with cyclic learning rate: `base_lr=5e-5`, `max_lr=2.5e-3`, `step_size=1000`

Thanks for reading.

2019.03.01