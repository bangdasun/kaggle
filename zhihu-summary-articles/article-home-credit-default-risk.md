## Kaggle竞赛-Home Credit Default Risk小结

竞赛网址: https://www.kaggle.com/c/home-credit-default-risk

### 1. Home Credit

Home Credit(中文: 捷信消费金融有限公司)是中东欧/亚洲的一家消费金融提供商, 为客户提供金融服务.

### 2. 比赛简介

#### (1) 业务背景

Home Credit希望能通过数据挖掘和机器学习算法来估计客户的贷款违约概率.

#### (2) 数据

Home Credit提供了多张数据表, 包括 

- (1) application_train/test 客户申请表

包含了目标变量(客户是否违约-0/1变量), 客户申请贷款信息(贷款类型, 贷款总额, 年金), 客户基本信息(性别, 年龄, 家庭, 学历, 职业, 行业, 居住地情况), 客户财务信息(年收入, 房/车情况), 申请时提供的资料等.

- (2) bureau/bureau_balance 由其他金融机构提供给征信中心的客户信用记录历史(月数据)

包含了客户在征信中心的信用记录, 违约金额, 违约时间等. 以时间序列(按行)的形式进行记录.

- (3) POS_CASH_balance 客户在Home Credit数据库中POS(point of sales)和现金贷款历史(月数据)

包含了客户已付款和未付款的情况.

- (4) credit_card_balance 客户在Home Credit数据库中信用卡的snapshot历史(月数据)

包含了客户消费次数, 消费金额等情况.

- (5) previous_application 客户先前的申请记录

包含了客户所有历史申请记录(申请信息, 申请结果等).

- (6) installments_payments 客户先前信用卡的还款记录

包含了客户的还款情况(还款日期, 是否逾期, 还款金额, 是否欠款等).

#### (3) 结果评判标准

比赛最终要求提交每个ID的违约概率, 以此计算得到的AUC作为评判标准.

### 3. 我的参赛方案

本次比赛的数据量不大, 训练集31万条, 测试集5万条, 但特征很多, 原始特征加起来超过200个, 而且分散在多个表. 所以首先要梳理清表的内容和表之间的关联, 从实际业务的角度来构造每个用户的特征. 整个Pipeline大致为: (1) 分析每张表的数据, 构建特征并保存; (2) 将所有表的特征通过Left Join到一起; (3) 构建表之间的交互特征; (4)训练模型. 下面简单介绍下我的参赛方案(130/7198).

#### (1) 特征工程:

- 原始特征: 

包括了原始特征和通过简单操作(加减乘除/分箱)构造的新特征, 比如客户收入和贷款总额的比率, 客户家庭人均收入, 客户年龄段, 客户收入段等.

- 统计特征:

除了申请表之外, 其他的表包含的单个客户数据都是一个时间序列, 因此可以对这个时间序列进行聚合(aggregation)求汇总统计量(mean/median/std/max/min/sum)等; 也可以根据分类变量(职业/行业/学历/年龄段/申请是否通过)等对数值变量(收入/贷款额/年金)进行聚合汇总, 作为一个"相对参考"的特征, 然后对客户原始数值和该参考数值做差得到差值特征. 

- 时序特征: 

只对一个时间序列求汇总统计量还是会损失不少信息, 从业务角度看, 客户的前几次申请和信用情况也非常重要. 所以可以按照 (1) 固定时间窗: 离本次申请最近的30/60/90/120天内的信用情况; (2) 固定次数: 本次申请的前1/2/3/5/10次申请情况来构造新特征, 继续采用汇总统计量.

- 弱模型特征:

这部分特征思路源于先前Avito比赛中的kernel: [LightGBM with Ridge Feature](https://www.kaggle.com/demery/lightgbm-with-ridge-feature), 即采用较弱模型得到out-of-fold预测值作为训练集特征, 预测值作为测试集特征, 属于模型Stacking. 我训练了Ridge/Logistic Regression(将Ridge的输出放到sigmoid函数中)/Factorization Machine/Neural Nets, 以及来自Scirpus的kernel [Pure GP with LogLoss](https://www.kaggle.com/scirpus/pure-gp-with-logloss)中的GP特征, 分别将这些模型的oof放入训练集, prediction放入测试集, 和别的特征一起重新训练新的模型.

#### (2) 模型:

- LightGBM/XGBoost: 最终融合了7-8个模型, 模型之间的差别在于预处理(是否移除缺失值/缺失值是否填补)/特征(不同的特征选择结果)/超参(自己调试的/来自kernel的baye化/Neptune-ml), 各个模型的cv在0.795-0.805之间, public lb分数在0.799-0.803之间, private lb分数在0.796-0.798之间. 
- Ridge/Logistic Regression/FM: 因为cv分数很低, 所以将这些模型的oof和submission放入了Boosting模型中作为stacking;
- FFNN: 参考kernel - [10 fold Ridge from Dromosys Features LB-0.768](https://www.kaggle.com/tottenham/10-fold-ridge-from-dromosys-features-lb-0-768), 同样cv不是很高, 所以将其oof和submission放入了Boosting模型.

最终融合的public lb分数最高为0.80556, private最高分数为0.79813.

#### (3) 训练与验证:

所有的单模都按5折和10折交叉验证进行训练, 对所有折取平均后进行提交. 这次比赛的cv和public lb并不十分一致, 一个原因在于public lb只保留了3位小数, 另一个原因在于public lb只有20%的数据. 相信cv是kaggle的第一原则, 我发现最后private lb最高的居然是一个单模, 而我当时没有将这个单模选入融合模型中. 在查看log之后发现该单模的cv并不是最好的, 且和其他几个单模有很高的相关性, 所以我没有选择它. 后来我分析可能原因在于我只计算了cv的均值, 忽略了标准差, 导致这一低级失误.


### 4. Top参赛方案

#### (1) 第1名 (Home Aloan)

原贴: [I am speechless](https://www.kaggle.com/c/home-credit-default-risk/discussion/64480)

**特征工程**: 分析了Neptune的open-solution, 添加了新的特征, 然后进行了特征选择.

**模型/训练和验证**: XGBoost/LightGBM/Catboost/Ridge/NN.

除此之外作者Bojan大叔主要分享了比赛的心路历程, 没有更多的比赛细节. 因此第1名的解决方案有待更新.

#### (2) 第2名 (ikiri_DS)

原贴: [2nd place solution ( team ikiri_DS )](https://www.kaggle.com/c/home-credit-default-risk/discussion/64722)

**特征工程**: 包括降维(PCA/UMAP/T-SNE/LDA), GP特征, 利率特征等. 团队成员较多, 细节可参考原贴.

团队成员Shuo-Jan Chang的分享: [HC - Brief solution from Shuo-Jen, Chang](https://storage.googleapis.com/kaggle-forum-message-attachments/379776/10237/HC%20-%20Brief%20solution%20from%20Shuo-Jen%20Chang.html)

**模型/训练和验证**: LightGBM(with dart), Catboost, DAE, CNN, RNN.

该团队做了一张很好看的模型blending示意图:

![2nd place Model Structure](https://cdn-ak.f.st-hatena.com/images/fotolife/g/greenwind120170/20180901/20180901083809.png)

团队的其中一个亮点在于Giba的发现(原贴: [Congratulations, Thanks and Finding!!!](https://www.kaggle.com/c/home-credit-default-risk/discussion/64485)): 他观察数据后发现可以识别到user_id(根据例如DAYS_BIRTH, DAYS_DECISION等一些用户自身属性), 在训练集和测试集中有8549个user有2行记录, 132个user有3行记录, 如果之前的申请记录目标变量为1, 则下一次申请有90%的可能也是1.

#### (3) 第3名 (alijs & Evgeny)

原贴: [3rd place solution](https://www.kaggle.com/c/home-credit-default-risk/discussion/64596)

**特征工程**: 成员Evgeny有在银行的工作经历, 因此构建特征主要从实际业务出发. 总共构建了1000个左右的特征, 选择了其中的250个, 特征选择通过cv来进行(纯手动), 模型训练速度很快, 所以短时间内就能完成特征选择. 对于bureau表和previous_application表, 作者没有根据当前ID进行聚合(接下来的解释暂时没有看懂, 原文为: I used data without aggregation by sk_id_curr and used target from main application (same for all data for one client). I was very weak model Predicts were averaged by sk_id_curr after. This approach performed better in main model compared to model with aggregated data with better individual score.), 希望之后可以问清楚具体细节. Evgeny还构建了可以预测EXT_SOURCES的模型, 预测值和残差值都是比较有用的特征(这个操作和Avito比赛中以商品价格预测值为特征是一样的), 他还尝试了预测收入, 但是没有用.

成员alijs主要构建了统计特征. 

**模型/训练和验证**: 在不同的数据表上建立了不同的模型. alijs训练了50个左右不同的模型, 选择了7个互相之间最不相关的模型而不是根据cv. 最后构建了4层的stacking, 包含了LightGBM, Random Forest, ExtraTree, Linear Regression.


#### (4) 第4名 (Quad Machine)

原贴: [4th place sharing and tips about having a good teamwork experience](https://www.kaggle.com/c/home-credit-default-risk/discussion/64487)

**特征工程**: 趋势特征, 在installment_payment/POS_CASH_balance/bureau表中最近的1/3/5/10条记录特征; 在使用bayes优化超参的时候将预测值保存下来作为oof; 在进行模型stacking之前进行了特征选择, 包括使用LightGBM的特征重要程度, Olivier的特征选择法(kernel见文末).

**模型/训练和验证**: 在不同的特征子集上训练模型, 总共训练了200个模型; 除此之外还进行后post-processing, 对于revolving贷款, 如果预测值超过0.4, 则乘以0.8, 因为revolving贷款在测试集中出现的较少, 这个比较trick, 作者认为这个方法不太会适用于别的场景.

团队成员Kain在讨论区回答了很多较为细节的问题, 请参考原贴. 下周该团队会把代码公布到github上.


#### (5) 第5名 (Kraków, Lublin i Zhabinka)

原贴: [Overview of the 5th solution +0.004 to CV/Private LB of your model](https://www.kaggle.com/c/home-credit-default-risk/discussion/64625)

**特征工程**: 构建了8000个特征, 最终基于Olivier的特征选择法保留了3000个. 将分类变量数值化(类似LabelEncoding); 各种统计groupby特征, 包括对时间窗进行groupby处理. 

**模型/训练和验证**: 该团队在模型构建方面有不少特别之处, 包括:

- NN:提取不同数据表之间的交互特征

来自不同表格的信息交互很难通过人为判断进行提取, 因此尝试用NN构建用户画像, 将其转化为用户分类问题. 在每张数据表上(application表除外)都可以构建一个用户的向量(每个月的数据), 然后将这些向量合并到一起, 得到一个较为稀疏的用户画像. 然后构建如下NN: (1) 对每个向量进行normalization: 除以最大值; (2) 输入为96个向量, 每个月为一个向量; (3) 1维卷积层(Conv1D); (4) Bidirectional LSTM; (5) Dense层; (6) 输出. 

这个模型cv为0.72, 加入stacking之后有0.001的提升.

- Nested模型

不同的groupby特征之间的关系很难理清. 例如5年前的违约记录肯定没有1个月前的违约记录来的重要. 因此构建了Nested模型. 以installment_payment表为例, 根据当前ID将application表中的目标变量加入到installment_payment表中, 然后训练一个LightGBM来预测每条记录对应的目标变量值, 可解释为"怎样的installment行为会对目标变量产生怎样的影响". 为此可以得到许多oof, 对这些oof按当前ID进行聚合可以作为新的特征.

接着作者还提到了对credit_card_balance表所做一些操作, 对cv有不小的提升, 但最后产生了过拟合, 因为贷款类型在训练集和测试集的分布差别较大.

- 利率/当前贷款的期限预测模型

这部分的理解需要一定金融知识, 待更新, 细节可见原贴.

- 银行系统打分模型

基于application/previous_application表构建了logit和probit模型, 对cv和public lb有一定的提升.

#### (6) 第7名 (A.Assklou _ Aguiar)

原贴: [7th solution - Not great things but small things in a great way.](https://www.kaggle.com/c/home-credit-default-risk/discussion/64580)

**特征工程**: 提取了500个左右的新特征(groupby特征), 有4个特征对模型有0.04的提升: NEW_CREDIT_TO_ANNUITY_RATIO, EXT_SOURCE_3, INSTAL_DAYS_ENTRY_PAYMENT_MAX, DAYS_BIRTH. 

**模型/训练和验证**: 采用了11个不同的LightGBM和2个NN, 最后用Linear Regression进行stacking, 所有的模型评判都以cv为准.


#### (7) 第8名 (七上八下)

原贴: [8th Solution Overview](https://www.kaggle.com/c/home-credit-default-risk/discussion/64474)

**特征工程**: 将所有表根据当前ID和时间融合到一起; 根据还款逾期天数(DPD)进行分箱处理(30/60/90/120); 将credit_card_balance表与application表融合后的大量NA用0进行填充. 在组队完成后, 将所有的特征放到一起训练一个单模, 同时发现将在单个表上训练得到的oof作为特征对cv和public lb都有提升.

**模型/训练和验证**: 最好的单模都是LightGBM. 采用Stratified 10折验证, local cv和private lb的走势十分一致. 最终的模型stacking框架图在讨论区给出, 每个团队成员都在单张数据表上训练了LightGBM/XGBoost/Random Forest/Logistic Regression, 输出oof, 再加上每个成员的全部特征集上训练输出oof, 用LightGBM(regressor)和Random Forest进行stacking.


#### (8) 第9名 (International Fit Club)

原贴: [#9 Solution](https://www.kaggle.com/c/home-credit-default-risk/discussion/64536)

**特征工程**: 成员CoreyLevinson, 没有在所有历史数据上进行groupby特征提取, 而是在最近的数据上进行(最近3/6/9/12/18/24/30个月); 替换测试集存在而训练集中不存在的类别; 对一些二分变量进行求和(FLAG_DOCUMENT_, REG_CITY_等); 比率特征; 将一些分类变量聚合到一起再进行Encoding等.

成员Suchith Mahajan, 通过对日期信息进行一定处理得到"multiplier", 然后应用到数值变量上, 用于反应时间信息. 细节可见原贴.

**模型/训练和验证**: LightGBM(dart), Entity Embedded NN(参考自Porto Seguro比赛), XGBoost, MICE imputation Model. 最后构建了一个使用200个模型的6层stacking, 使用Logistic Regression作为最后的stacker.


#### (9) 第10名 (Best Friend Forever: CV)

原贴: [10th place writeup](https://www.kaggle.com/c/home-credit-default-risk/discussion/64598)

**特征工程**: 利率特征, 现金贷款模型: 根据实际金融业务知识构建特征; 训练LightGBM来预测EXT_SOURCE_1和EXT_SOURCE_3, 因为EXT_SOURCE_有较多的缺失值.

**模型/训练和验证**: LightGBM等. 在stacking时根据cv来选择最终提交结果, 使得最后private lb的分数进入了金牌区.

#### (10) 第12名 (楼上神仙打架 ¯\_(ツ) _/¯)

原贴: [#12 solution](https://www.kaggle.com/c/home-credit-default-risk/discussion/64504)

**特征工程**: 统计特征, 包括加减乘除和各种groupby, 以及使用时间窗划分; 模型特征, 使用NN和LightGBM在每张表上进行训练: 使用RNN和CNN在历史贷款的基础上预测未来贷款的情况, 使用LightGBM来预测该历史贷款是否来自于违约用户.

**模型/训练和验证**: 包括了blending(融合)和stacking. 随机选取一半的特征构建模型, 再采用不同的随机种子得到不同结果, 最终对所有结果进行融合. 


#### (11) 第13名 (KazAnova&Dmitry [h2o.ai])

原贴: [13th place - time series features](https://www.kaggle.com/c/home-credit-default-risk/discussion/64593)

**特征工程**: 总共构建了10000个左右的特征, 有1000个特征是和Neptune的open-solution基本一致的. 剩下的特征主要是时间序列特征: 设置lag为96, 用SQL构建了包括对bureau status/credit/credit balance/installment按月求和等特征, 总共有60个时间序列, SQL代码可参考原贴. 在这个时间序列的基础上, 作者进行了更多的操作: 根据不同的lag求移动平均, 使用max/min等更多的汇总函数, 指数平滑, 回归预测等. 作者在帖子中给出了几个相关链接以供参考. 

**模型/训练和验证**: 总共训练了100个模型, 包括LightGBM, XGBoost, CNN, LSTM等. 采用了两种验证方式, 一种是简单平均, 另一种是重新在全部数据上进行训练来得到预测值.


#### (12) 第14名 (seagull, Adri and Tomoyo)

原贴: [#14 solution](https://www.kaggle.com/c/home-credit-default-risk/discussion/64502)

**特征工程**: 对application/bureau/previous_application中的分类变量采用Target-Encoding, 使用max/min/mean. 对除了application之外的所有表格进行了时间窗划分, 然后用LightGBM输出的特征重要性进行特征选择(删去40%的特征). 

**模型/训练和验证**: 使用不同的特征集训练了包括Dart/Goss/Catboost/XGBoost在内的多个模型, 然后用LightGBM和ElasticNet作为stack模型, 融合了多个随机种子运行的结果.


#### (13) 第16名 (wl_team)

原贴: [The 16th Solution](https://www.kaggle.com/c/home-credit-default-risk/discussion/64505)

**特征工程**: 开始是暴力构造各种groupby特征, 然后根据时间窗(最近的6/12个月)进行特征选择, 将cv从0.785提升至0.795. 在installments_payment表中计算了各个还款日期直接的时间差, 对cv也有一定的提升. 然后根据XGBoost的特征重要程度选择了前400个特征, 加上Scirpus的GP特征, 但发现加入之后会过拟合. 因此将这些特征随机分为4部分, 然后选择加入或不加入GP特征, 总共得到8个不同的训练集.

**模型/训练和验证**: 对每个训练集训练一个LightGBM和XGBoost, 融合这16个模型之后cv为0.810, private lb为0.802. 


#### (14) 第17名 (qchemdog)

原贴: [17th place mini writeup](https://www.kaggle.com/c/home-credit-default-risk/discussion/64503)

这是排名最高的solo金牌.

将先前的历史申请作为训练数据, 目标变量为当前申请的目标变量. 因此在previous_application表的基础上, 加入其它表的特征(根据历史申请ID作为key而不是当前申请ID). 作者认为这样可以提取到历史申请和当前申请之间的相关关系. 在得到预测值后可以根据当前申请ID进行聚合汇总(mean/max/sum), 这些聚合特征是非常好的特征. 在对bureau表进行同样的操作后, 将cv从0.798提升到0.801.

在评论区也有人提到有相同的操作, 但是baseline是Neptune的open-solution, 所以最后过拟合了...

![facepalm](http://apatheticthursday.net/content/3-projects/201707-pathetic-thursday/Facepalm.png)


#### (15) 第19名 (AllMight)

原贴: [# 19th solution](https://www.kaggle.com/c/home-credit-default-risk/discussion/64592)

**特征工程和特征选择**: 融合所有表的数据后, 按一定时间窗(最近1/2/3年)对数据进行划分, 接着再将数据按申请是否被批准/信用卡是否关闭进行划分. 然后进行特征选择, 选取部分特征训练一个LightGBM, 根据特征重要程度选前50个重要特征, 再从剩下的特征中随机选取50个. 

**模型/训练和验证**: 对约200个模型进行stacking, 用了两层stack net: 第1层为LightGBM/XGBoost/Random Forest/Adaboost/Logistic Regression/ExtraTree, 第2层为LightGBM. Stacking之后的cv为0.803. 在训练LightGBM时metric采用binary_logloss时cv要高于auc.


#### (16) 第24名 (Abdel _ Arthur)

这是金牌区的最后一位.

原贴: [24th place - Simple Solution with 7 Models.](https://www.kaggle.com/c/home-credit-default-risk/discussion/64548)

**特征工程**: 总共构造了7个不同的特征集用于不同的模型. 包括了aggregated/binarized/statistics/lags等特征, 细节待更新.

**模型/训练和验证**: 共训练了6个Boosting模型, 每个模型都使用bayes优化得到最佳超参; 1个NN模型, 使用PReLU/BatchNorm/High Dropout. 最后通过一个Logistic Regression进行stacking. CV较为准确.


#### (17) 第27名 (nyanp)

原贴: [Pseudo Data Augmentation (27th place short writeup)](https://www.kaggle.com/c/home-credit-default-risk/discussion/64693)

作者根据目标变量的定义构造了一个pseudo target. 目标变量的原始定义为: 客户在前Y次付款中至少有一次有超过X天的延期付款, 可以通过installments_payments表中的延期付款天数(DPD)进行估计, 作者在原贴中给出了源代码, 细节可参考原贴. 在得到这个pseudo target后, 他在历史申请数据上进行了训练, 得到的预测值作为预测当前申请的特征.

作者的源代码已发布到github: [nyanp/kaggle_homecredit](https://github.com/nyanp/kaggle-homecredit).


#### (18) 第32名 (arnowaczynski)

原贴: [Story behind the 32nd place (top 1%) with 2 submissions](https://www.kaggle.com/c/home-credit-default-risk/discussion/64609)

只用了两次提交, 经过正确的调参后(主要是num_boost_round, learning_rate, num_leaves, max_depth, min_date_in_leaf, feature_fraction, bagging_fraction, bagging_freq)融合了30个LightGBM模型和60个Ridge模型: 不同的折数(大部分是10折)和不同的随机种子.


最后列一些我参考过和认为不错的kernel:

- [Visualizing User_ids](https://www.kaggle.com/titericz/visualizing-user-ids)
- [A competition without a leak. Or is it?](https://www.kaggle.com/raddar/a-competition-without-a-leak-or-is-it)
- [10 fold Simple DNN with Rank Gauss](https://www.kaggle.com/tottenham/10-fold-simple-dnn-with-rank-gauss)
- [Pure GP with LogLoss](https://www.kaggle.com/scirpus/pure-gp-with-logloss)
- [10 fold Ridge from Dromosys Features LB0.768](https://www.kaggle.com/tottenham/10-fold-ridge-from-dromosys-features-lb-0-768)
- [LighGBM_with_Selected_Features](https://www.kaggle.com/ogrellier/lighgbm-with-selected-features)
- [Simple Bayesian Optimization for LightGBM](https://www.kaggle.com/sz8416/simple-bayesian-optimization-for-lightgbm)
- [Feature Selection with Null Importances](https://www.kaggle.com/ogrellier/feature-selection-with-null-importances)
- [Entity Embedding Neural Network Keras LB 0.748](https://www.kaggle.com/astrus/entity-embedding-neural-network-keras-lb-0-748)

以及Neptune的open solution:
- [neptune-ml/open-solution-home-credit](https://github.com/neptune-ml/open-solution-home-credit)

### 5. 总结

最终排名130/7189, 按百分比算这次比赛是成绩最好的一次了. 本次比赛的特点在于特征较多, 表与表之间的关系也较为复杂, 但可以只用GBM类的模型就取得不错的成绩. 从这些金牌区的分享来看, 有依靠出色的数据洞察力的(比如第2/5/17名), 也有依靠业务知识+模型创新的(第5名), 当然也有万年不变的stacking大法 -- 训练几百个模型来搭一通积木; 但共同点是显然的, 有靠谱的模型验证策略, 以及信任本地cv的结果, 正是这一条, 保证了这些队伍在private lb有较大shake up的情况下获得了金牌, 或者只通过个位数的提交取得前1%的成绩.

作为上班族, 能做比赛的时间只有每天晚上的1-3小时和周末, 过了一遍这些优胜方案之后, 觉得自己要取得好成绩还是先得加大时间的投入, 下次比赛尽量早点加入, 尽可能多的对数据进行分析. 这次还有一个小插曲, 就是第10名队伍中一位成员在merge ddl之前发出邀请, 结果发现我提交次数太多, 和他们的提交总数加起来已经超过上限, 无奈只能继续solo. 我有时候还是按捺不住想多提交几次, 说到底还是Too young, 所以今后比赛在以好名次为目标的同时, 也要放平心态, 相信自己的local cv.
