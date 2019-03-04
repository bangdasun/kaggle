## Kaggle竞赛-Elo Merchant Category Recommendation小结

竞赛网址: https://www.kaggle.com/c/elo-merchant-category-recommendation

发现已经有两篇参赛方案分享在知乎了:

- [kaggle ELO competition 22nd solution](https://zhuanlan.zhihu.com/p/57969578)
- [kaggle Elo 27th top1% solution-feature engineering](https://zhuanlan.zhihu.com/p/57815923)

### 1. Elo

Elo是一家巴西的提供支付服务公司.

### 2. 比赛简介

#### (1) 背景

Elo与许多商户有合作关系, 商户希望可以对特定用户制定优惠策略和折扣, 也就是个性化推荐. Elo内部有一个业务指标, 称为loyalty score. Elo希望能够通过一系列历史数据对每个用户(每张卡)的loyalty score进行预测, 以提高推荐的精准度.

对于赛题背景的理解, 还可以参考: [Maybe we should take attention on Category INFO](https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/75034#442077)

#### (2) 数据

主要用到的有6张数据表:

- historical_transaction

历史交易数据, 包含的是训练和测试集中卡的交易数据, 所有的卡都有至少3个月交易记录. 包括卡ID, month_lag(相对于reference日期的月份数, 这里不太好翻译), 交易日期purchase_date, 交易是否成功authorized_flag, 分期购买数installments, 商户种类merchant_category_id, 商户大类subsector_id, 商户ID, 交易金额purchase_amount, 交易城市和州以及3个匿名分类变量(category_1, category_2, category_3)

- new_merchant_transaction

新商户交易数据, 包含的商户是用户在两个月内第一次消费的. 特征和historical_transaction表中一致. 和historical_transaction表的区别在于, 新交易的数据month_lag为1或2, 历史交易的数据month_lag为-13至0. 根据month_lag的定义, month_lag为0时所在的交易日期就是所谓的reference日期.

- merchants

商户信息. 主要是商户的流水信息, 比赛中并没有太大帮助.

- train

训练集. 包括了卡的ID, 卡第一次被激活的年月(first_active_month), 和3个匿名分类特征(feature_1, feature_2, feature_3), 还有目标变量loyalty score.

- test

测试集, 除了没有目标变量以外和训练集一样.

#### (3) 结果评判标准

以预测的loyalty score与真实值的RMSE(Root Mean Squared Error)作为评价指标.

### 3. 我的参赛方案

典型的特征赛, 是个检验特征赛各类套路的好机会.

#### (1) 特征工程

这次我主要的关注重点就是特征工程. 从表的结构和表之间的关系来看, 需要提取的是每张卡(训练集中对应的card_id)的特征, 而在最重要的两张表historical_transaction和new_merchant_transaction中, 每张卡都有多条交易记录, 所以可以从聚合(aggregation)卡的各种记录信息开始入手. 我按特征的业务意义对特征进行分类: 基本信息特征, 金额特征, 时间特征. 这是根据客户关系管理中衡量客户价值的RFM(Recency - Frequency - Monetary)模型进行划分的.

- 基本信息特征
  - 每个card_id的交易次数(count)/成功和失败次数(按authorized_flag分类)
  - 每个card_id在不同city_id/state_id/merchant_category/subsector_id/merchant_id下的unique count/max/entropy/
  - 每个card_id在不同category_1/category_2/category_3下的count/mean/std/entropy, 这里首先对3个分类变量进行one-hot转化
  - 每个card_id在不同month_lag下的count/max
  - 每个card_id在相同merchant_id下再次交易的count
  - ...
- 金额特征
  - 每个card_id交易金额的sum/mean/max/min/median/std/skew
  - 每个card_id在不同month_lag下金额的sum/mean/max
  - 每个card_id经过时间衰减加权(time decay weighted)后交易金额的sum/mean/max/min/median/std/skew
  - 每个card_id的交易金额序列(按时间排序)降维特征, 类似于对文本Tfidf特征矩阵进行PCA降维
  - 按其他分类变量进行分组(groupby)后提取各种统计特征
  - ...
- 时间特征
  - 每个card_id第一次和最后一次交易的时间和两者的时间差
  - 每个card_id交易时周末和工作日的比例
  - 每个card_id交易日期和固定日期(2018-12-31)时间差的mean/std/max/min
  - 每个card_id近n个月的交易次数/金额统计特征
  - 每个card_id交易时间前后时间差的mean/std/max/min
  - 每个card_id在节假日的交易统计特征
  - ...

其中也包括对一些特征进行简单四则运算构造交互特征(历史交易和新交易之间的交互). 对于historical_transaction, 我根据authorized_flag进行划分, 所以对于历史交易数据有全部数据和成功交易数据两部分. 总体来说没有特别tricky的特征, 都是常规方法, 只要仔细的对数据进行探索性分析, 搞清大致业务逻辑, 都可以想到提取这些特征. 最后我提取了807个特征, 这里不进行更仔细的讨论. 细节可以看我放在github上的代码.

#### (2) 模型

全部是lightgbm模型. 讨论区有不少人使用NN或者FM, 因为时间原因我没有进行尝试. 

#### (3) 训练与验证

采用5折交叉验证进行训练, 上述所有特征经过不同特征选择(lightgbm在不同的特征集上训练)后简单加权融合结果的CV可以到3.630左右. 而我发现论坛里许多前排的参赛者说CV在3.64以上, 这还是让我挺有安全感的, 虽然我public LB还在3.685左右.

我参考了public kernel的做法, 将异常值(-33左右, 占训练集的1%)剔除后在正常数据上训练模型, CV在1.546左右, 可见这1%的异常值贡献了大部分的误差. 所以再训练异常值分类模型, 这里有两种用法: (1) 将异常值预测概率较低的数据用正常数据训练出来得到的预测值进行代替; (2) 将异常值预测概率较高的数据直接赋值为-33. 这两种方法都有一定的风险, 第一种方法中过高的False Negative和第二种方法中过高的False Positive都会使结果更差. 这与异常值预测概率的阈值有关, 阈值的选择应根据CV确定.

我在最后一周半的时候开始异常值检测的工作, 我分别尝试将异常值预测概率最高的前几个数据赋值为-33, public LB有提升, 但这只能说明检测到的异常值在public LB, private LB的分数变化是完全不知道的, 这样的工作在我看来没有太大的意义.

我最终两次提交分别为public LB 0.670/private LB 0.611(对预测的异常值直接赋值-33)和public LB0.680/private LB0.602(不处理预测的异常值), 后者的public LB在300名左右, 最后private LB排名28, 说明跟着CV提升的方向走大致没有问题, 同时说明异常值分类算法的False Positive太高了. 

金牌线的分数为3.600, 或许我多训练几个模型增加点多样性就能达到了, 略遗憾 ;)

最后放一个我的github, 我简单整理了比赛所用的代码: https://github.com/bangdasun/kaggle/tree/master/elo-merchant-category-recommendation. 

### 4. Top参赛方案

#### (1) 第1名(Look alive)

第1名是[@砍手豪](https://www.zhihu.com/people/kan-shou-hao/activities)大佬! 相信许多人都看过他的知乎文章: [零基础自学两月后三月三次轻松进入kaggle比赛top20小结](https://zhuanlan.zhihu.com/p/29923137), 这次比赛仅用24次提交就拿到solo冠军, 太强了, 期待他能在知乎上进行后续的分享. 他暂时在kaggle讨论区分享了对于异常值的处理方法.

原贴: [My simple trick for this competition](https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/82036)

最后的预测值为:

```python
train['final'] = train['binary_predict'] * (-33) + (1 - train['binary_predict']) * train['no_outlier']
```

这里的`train['binary_predict']`是异常值分类模型预测概率, `train['no_outlier']` 是仅在正常数据上的预测值. 十分直观合理的方法, 使得local CV下降了0.015. 而分类模型的特征参考了Home Credit比赛中最高solo金牌的做法: [17th place mini writeup](https://www.kaggle.com/c/home-credit-default-risk/discussion/64503#378162).

我尝试了一下这个方法, 但是分数还是不如最好的提交, 可能是因为异常值分类模型不够好(CV AUC = 0.9089, 砍手豪的AUC = 0.914). 但对于这个思路, 真的是..



给跪了.



#### (2) 第5名([ods.ai] Evgeny Patekha)

原贴: [#5 solution](https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/82314)

**特征工程**: 使用了一些target encoding特征, 在计算时进行了时间衰减加权的处理, 即赋予较小的month_lag较小的权重. 总共提取了1000+的特征, 经过特征选择后保留了100+个特征.

**模型**: 训练了:

- 全部数据上的回归模型

- 异常值分类模型, 按预测值(阈值0.015)划分了训练集和测试集来训练低概率和高概率模型. 预测概率同时也作为特征用来训练高概率模型

最后融合的是低概率模型/高概率模型和全部数据上训练的回归模型的预测值.

#### (3) 第7名(You'll Never Overfitting Alone)

[@senkin13](https://www.kaggle.com/senkin13)大佬也是很强, 尤其是特征比赛, 每次都让人感觉稳的不行. 他的分享方案十分详细.

原贴: [7th Place Solution (Updated Late Submission Finding)](https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/82055)

**预处理**: 首先对historical_transaction和new_merchant_transaction按时间进行排序, 新生成8张数据表:

- 只有历史交易数据(historical_transaction)
- 只有历史成功交易数据(historical_transaction, authorized_flag=1)
- 只有新交易数据(new_merchant_transaction)
- 融合历史成功交易数据和新交易数据
- 融合历史交易数据和新交易数据
- 融合历史交易数据和商户数据(merchant)
- 融合新交易数据和商户数据
- 融合历史交易数据, 新交易数据和商户数据

**特征工程和特征选择**:

- 聚合特征, 即groupby + 统计函数
- 时间区间特征: 交易时间的时间差
- 交互特征: 新交易数据的最后一次交易时间 / 历史交易数据的最后一次交易时间
- SVC特征: 每个card_id的merchant_id/merchant_category序列Tfidf矩阵降维特征
- Word2Vec特征: 提取每个merchant_id/merchant_category/purchase_date的词嵌入特征, 然后按card_id计算min/max/mean/std
- meta特征: 采用第6,7,8张数据表, 将loyalty score加入表中训练模型, 预测值作为新的特征, 然后按card_id得到min/sum. 这些特征对于CV和public LB都有0.005-0.006的提升.

特征选择方面, 采用了Home Credit比赛中计算Null importance的方法.

**模型**: lightgbm和3层的MLP. 总共训练了12个lightgbm和40个NN模型进行stacking, 但最终private LB结果不如最好的单模. 使用Isotonic回归, CV和public LB都有0.005-0.006的提升, 但是会导致过拟合使得private LB变差.

#### (4) 第10名([ods.ai] YuryBolkonskiy)

原贴: [10th place solution](https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/82093)

**预处理和特征工程**: 将历史交易成功数据和新交易数据融合. 特征工程方面:

- 对merchant_id/subsector_id/merchant_category采用CountVectorizer, 然后用PCA和SVD降维
- 按month_lag分类, 提取交易次数/交易金额的统计特征, 同时提取不同month_lag之间的交互特征
- 对不同month_lag下的交易金额进行简单移动平均和指数平滑处理
- 计算交易的时间差统计特征

**特征选择**: 总共生成了6500+个特征, 选择方法:

- Boruta方法(8小时选择500个最佳特征)
- 通过lightgbm/xgboost/catboost的特征重要性选择特征
- 根据Adversarial validation删除特征

**模型**: 最佳单模为5折catboost. 还使用了DeepFM和lightgbm. 最后融合了47个模型(训练时包括异常值)和35个模型(训练时剔除异常值).

#### (5) 第11名(Stack It All)

原贴: [11th place solution](https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/82127)

**特征工程**: 和public kernel一样生成了许多聚合特征, 作者在帖子里列出了较强的特征. 

**特征选择**: 根据特征重要性和相关性进行特征选择, 其他还包括:

- 删除缺失值比例太高的特征
- 方差太小的特征

**模型**: lightgbm/catboost/xgboost/h2oRF/h2oGBM. 总共融合了32个模型, 使用贝叶斯回归.

**后处理**: 和我的做法一样, 最后选择前21个异常值概率最高的数据作为异常值. 最后对于private LB也有帮助.

作者还分享了一些不成功的尝试, 细节可见原贴.

#### (6) 第16名(nlgn)

原贴: [16th Place Summary](https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/82166)

**特征选择**: 根据[Reducing the gap between CV and LB](https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/77537)中的描述移除了训练集和测试集中分布不一致的特征, 同时也采用了Home Credit比赛中Null importance的特征选择方法.

**模型**: 和我的做法基本一致, 最后融合: 全部数据的回归模型/后处理后的回归模型和异常值概率预测值, 使用贝叶斯回归得到最后的结果.

#### (7) 第18名(是風的追求還是樹的不挽留)

原贴: [18th place story (pocket's side)](https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/82107)

作者[@pocket](https://www.kaggle.com/pocketsuteado)分享了他负责的部分.

**特征**: 

- 基本的聚合特征
- 时间特征: 最后一次交易的日期是比较强的特征
- 对authorized_flag = 1和city_id = -1进行划分; 强特征
- 对month_lag >= -2和month_lag = 0进行划分; 强特征
- 预测最后一次交易日期, 然后计算其与真实值的差值
- 对交易小时进行分箱处理
- target-encoding

**模型**: 使用简单的线性回归和岭回归(ridge regression)进行stacking, 效果比lightgbm要好.

作者还列出了一些被删除的特征和模型, 细节可见原贴.

#### (8) 第21名(Trust Your CV)

原贴: [21th place solution](https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/82235)

作者较为简单地分享了一些要点, 其代码已在github开源: https://github.com/bestpredicts/ELO

#### (9) 第22名(Grand Rookie)

原贴: [22nd (some 10k feats)shareing some engineering and some trick(not used)](https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/82057)

作者提到他们有类似于第1名trick的方法, 但是没有选择将其放入最后的提交, 失误了. 总共生成了1000+个特征. 作者分享了其中的一些, 还有lightgbm的特征重要性.

还可以参考他们的知乎文章: [kaggle ELO competition 22nd solution](https://zhuanlan.zhihu.com/p/57969578)

#### (10) 第27名([Datawhale] Next in money)

可以参考他们的知乎文章: [kaggle Elo 27th top1% solution-feature engineering](https://zhuanlan.zhihu.com/p/57815923)

#### (11) 第28名(bangda)

正是在下:), 参考前一章节.

除此之外还有一些参赛方案的分享:

- [19th place memo](https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/82178)

- [31st place, Some of my Feature Engineering](https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/82084)
- [32nd Place - Barely shaken up - It was all about diversity](https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/82126)
- [55th solution sharing](https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/82062)

最后列一些我参考过和认为不错的kernel, 这次kernel区没有强的单模, feature engineering方面都是常规方法. 相比之前一些特征赛, 各路blend党也消停了不少:

- [target - true meaning revealed!](https://www.kaggle.com/raddar/target-true-meaning-revealed)
- [card_id loyalty - different points in time](https://www.kaggle.com/raddar/card-id-loyalty-different-points-in-time)
- [Towards de-anonymizing the data! Some insights](https://www.kaggle.com/raddar/towards-de-anonymizing-the-data-some-insights)
- [Combining your model with a model without outlier](https://www.kaggle.com/waitingli/combining-your-model-with-a-model-without-outlier)

### 5. 总结

从前排分享的结果来看, 除了砍手豪的trick外, 特征工程 + 模型融合还是取得好成绩的关键. 而从最后结果来看, 我的特征工程和特征选择还是比较有效的, 我发现最好的单模已经可以达到金牌区选手的单模水平, 差别还是在模型融合上面. 对于solo选手, 模型的多样性还是会弱一点. 希望下次有机会可以组队获得更好的成绩.

最后, 一天连着写了两篇总结, 还是有点累的..但确实学到不少 :)