### Elo Merchant Category Recommendation

#### Competition Introduction

See my zhihu article: [Kaggle竞赛-Elo Merchant Category Recommendation小结](https://zhuanlan.zhihu.com/p/58231255)

#### About My Solution (28/4128)

This is my current best rank across all competitions I've participated. The key to get this rank is trust CV with feature engineering and feature selection. The evaluation metric is RMSE, and there are around 1% outliers in training data which contributes the most RMSE (with outlier CV RMSE around 3.625, without outlier CV RMSE around 1.547). Therefore many people did outlier detections and manually set the detected outliers to be outlier value in training data. This could get low RMSE score on public LB, but it is risky since the improvement only indicates that outliers in public LB part are being detected, the private LB could be either better or worse.

I spent about 1 month on feature engineering and feature selection, last 1 week and a half I started outlier detection stuff which could improve the LB score by 0.01, but finally the submission without outlier detection won. And I saw many people are "shaked down", might because they got too many false positives in outlier detection on the private LB part.

For feature engineering, there is nothing fancy, I started from my previous experience and some business insights. Feature selection helped a lot for me, I first generate as many features as I can, then do backward selections. My final 2 submissions:

- blend my best 6 models (all lightgbm) with 2 public kernels (I did some improvements on them); build outlier classification model and train lightgbm on normal data only, set top k lowest outlier probability samples' target to be predictions from models trained on normal data, probability threshold is based on CV score
- apply post-processing: set top k highest outlier probability samples' target to be -33 based on submission (1)

A general running instruction for my model (open issues here or contact me if you have questions):

##### (1) install a customized package contains some utility functions

Run:
```
git clone https://github.com/bangdasun/kaggle_learn.git
```
within the work directory.

##### (2) download data from competition website: https://www.kaggle.com/c/elo-merchant-category-recommendation/data

##### (3) run preprocessing scripts

Run:
```
python preprocess_hist_transac.py
python preprocess_new_transac.py
```
##### (4) run feature extraction scripts

Run:
```
python feature_extraction_hist_transac_info.py
python feature_extraction_hist_transac_amount.py
python feature_extraction_hist_transac_time.py
python feature_extraction_hist_transac_info_a.py
python feature_extraction_hist_transac_amount_a.py
python feature_extraction_hist_transac_time_a.py
python feature_extraction_new_transac_info.py
python feature_extraction_new_transac_amount.py
python feature_extraction_new_transac_time.py
python feature_extraction_merchant_repurchase.py
```
Besides these, there are also some categorical embedding features, remember the output features file is currently hard coded, therefore you need to change it before each time you run the script.
```
python feature_extraction_categorical_lda_decomp.py merchant_category_id 1 300 10
python feature_extraction_categorical_lda_decomp.py merchant_category_id 2 300000 50
python feature_extraction_categorical_lda_decomp.py merchant_id 2 300000 50
```
these will generate:
```
hist_transac_merchant_category_lda_comp_0.csv
hist_transac_merchant_category_lda_comp_2.csv
hist_transac_merchant_id_lda_comp_0_1.csv
```

##### (5) run feature selection

Run:

```
python run_lgbm_w_feats_selection.py
```
which should output 5 submissions.

##### (6) final blends

Blend submissions.

In the end, I have my collection of helper function which I will use, it is called `kaggle_learn`, here is the repo [bangda/kaggle_learn](https://github.com/bangdasun/kaggle_learn).

Thanks for reading.

2019.03.02