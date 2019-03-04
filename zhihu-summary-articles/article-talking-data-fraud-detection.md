## Kaggle竞赛-TalkingData AdTracking Fraud Detection Challenge小结

竞赛网址: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection

### 1. TalkingData

TalkingData是一家总部位于北京的第三方数据服务提供商, 致力于用数据改变企业决策. 其智能数据平台覆盖超过6.5亿/月活跃独立智能设备, 客户包括了互联网行业企业(谷歌, 腾讯等)和传统行业企业(银联, 招商银行等).

### 2. 比赛简介

#### (1) 业务背景

比赛要求参赛者解决一个二分类问题, 即预测用户在点击移动端关于一些App的广告后是否会下载该App; 这也是很多互联网公司所关注的"转化率"预估问题. 从另一个角度看, 在广告商投放广告之后, 广告平台方可以通过刷流量的方式来增加点击量, 但却不下载App. 因此虽然流量很高, 但都是虚假流量, 转化率很低, 并不能为广告商增加实际收益. 因此这次比赛也可以理解为防止刷流量搞欺诈, 这也是TalkingData的一块业务, 相关介绍可参考[科考作弊与广告作弊怎么可以这么像！](https://mp.weixin.qq.com/s/y83E4kfm-yiwkYKBxysyeg?)

#### (2) 数据
TalkingData提供了样本量近1.9亿的训练数据, 包括了2017.11.06至2017.11.09之间的数据, 每条数据记录为一次广告点击, 包括以下字段:

- ip: 点击的ip地址;
- app: 广告商提供的app id;
- device: 用户的移动设备id, 如iphone 6, iphone 7;
- os: 用户移动设备的操作系统版本id;
- channel: 广告投放渠道id;
- click_time: 点击时间(UTC时间), 格式为yyyy-mm-dd hh:mm:ss;
- attributed_time: 若用户下载了app, 这就是下载时间;
- is_attributed: 是否下载了app, 这是目标变量;

这里ip, app, device, os, channel都是分类变量且经过编码处理. 

#### (3) 结果评判标准

比赛最终要求提交用户下载app的概率, 以此计算得到的AUC作为评判标准.

### 3. 我的参赛方案

这是一次典型的数据挖掘类型比赛, 也是难得的一次不需要NN就可以取得好成绩的比赛, 因为比赛的重点在于特征工程. 接下来简要总结下我所尝试的特征和模型. 

#### (1) 特征工程:

- 原始特征: ip, os, app, channel, device, 以及通过click_time得到的day, hour, min, sec, 将他们作为分类特征进行处理. 我一开始尝试了one-hot特征, 发现维数过高, 且表现一般
- 统计特征: 通过不同的分类变量groupby组合, 计算某些变量的count, nunique, mean, var等特征
- 时序特征: 通过不同分类变量groupby组合, 计算下一次/上一次点击的间隔(next_click, prev_click)等特征; 计算连续若干次点击的间隔(time-delta)特征.

#### (2) 模型:

- logistic regression: 只在一开始使用了一下, 效果较为一般
- lightgbm: 本次比赛主打模型
- fm_ftrl: 主要基于该[kernel](https://www.kaggle.com/anttip/talkingdata-wordbatch-fm-ftrl-lb-0-9769)

#### (3) 训练与验证:

在11.08或11.07和11.08上进行训练, 在11.09上进行验证. 因为有人猜测并验证了public的数据全部为4点的数据, 所以可以将验证集分为两部分, 其中一部分作为public lb的模拟, 另一部分作为private lb的模拟. 同时对于测试集的特征通过test supplement计算得到. 最终融合(简单加权平均)了2个lightgbm和fm_ftrl模型, public lb的分数为0.9804, private lb的分数为0.9809. 

### 4. Top参赛方案

难得有次赛后有这么多top方案开源, 本文重点来了.

#### (1) 第1名 (['flowlight', 'komaki'].shuffle())

原贴: [1st place solution](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56475)

**特征工程**: 5个原始分类特征(ip, os, app, channel, device), 2个时间分类特征(day, hour), 若干统计特征(groupby-count). 接着尝试了原始5个分类特征所能得到的所有groupby组合(2^5-1=31), 对于每个groupby, 选取以下特征:

- 接下来1小时和6小时的点击数(count)
- 计算前向和后向的click time-delta
- 历史点击的平均下载率

将全部特征都放入模型, 此时模型可以在public lb上达到0.9808. 

接着对于分类变量的组合(总共20种)尝试用LDA/NMF/LSA得到嵌入(embedding)特征, 将n_component设置为5. 这样每种方法可以得到100个特征, 最后得到了300个特征(LDA/NMF/PCA), lightgbm分数达到0.9821. 实现这一方法的伪代码可见原贴.

之后, 移除除了app之外的所有分类变量, 这使得lightgbm分数达到了0.9828.

**模型**: 采用多个lightgbm和NN. 其中NN主要来自于公开的kernel, NN的更多细节可参考原贴.

**训练与验证**: 进行负采样, 选取全部的正样本(is_attributed==1), 然后选取相同样本量的负样本, 这意味着抛弃99.8%的负样本. 发现模型的表现并没有受太大影响 (特征工程在全部数据上进行抽取而不是仅在抽样之后). 同时, 采样不同的负样本, 然后进行融合有不错的提升 (即训练5个模型, 每个模型采样不同的随机种子), 同时也使得模型训练时间大大降低. 选取11.07和11.08的数据进行训练, 11.09的数据进行验证, 得到迭代次数等参数后重新在11.07-11.09的数据上进行训练. 总共有646个特征, 5个lightgbm融合后的分数为0.9833(public lb)和0.9842(private lb). 最后的提交方案为rank-based加权平均, 融合了7个lightgbm和1个NN, public lb得分为0.9834.

#### (2) 第2名 (PPP)

原贴: [[2nd Place Solution] from PPP](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56328)

**特征工程**: 没有magic特征, 主要特征和kernel里的基本一致(较早得到了不错的特征, 但后来大家都挖掘出这些特征了), 通过groupby构造特征(包括count, cumcount, unique-count, time-delta), 还有"which [app/os/channel]s each ip appears in data"的特征, 即计算每个ip在某些app/os/channel上的点击数(选取点击频率最高的几个). 总共尝试了上百个特征.

**模型**: 采用了lightgbm和neural nets; 最佳单模为lightgbm, private lb分数为0.9837; 最佳NN的private lb分数为0.9834 (对分类变量采用dot-product layers, 对连续变量采用fully-connected layers). 每个人都训练了一个lightgbm和NN, 总共有6个模型, 对6个模型的结果直接采用简单加权平均.

**训练与验证**: 对负样本进行向下抽样; 在全部数据上进行特征抽取, 然后和抽样得到的样本进行融合. 采用5折交叉验证来进行线下模型评价.

#### (3) 第3名 (bestfitting)

原贴: [My brief summary,a mainly NN based solution(3th)](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56262)

bestfitting目前kaggle competition总排名第一, 非常厉害.

**特征工程**: 主要采用的有23个特征, 其中比较重要的是点击之间的时间差(time-delta)特征, 抽取了每个点击的前5次与后5次点击之间的时间差作为特征. 除了原始数据中的channel, os, app, device特征外, 还有点击小时(hour)和统计特征, 所有特征根据特征重要性从大到小为:

- channel                                  1011
- os                                        544
- hour                                      472
- app                                       468
- ip_app_os_device_day_click_time_next_1     320
- app_channel_os_mean_is_attributed         189
- ip_app_mean_is_attributed                 124
- ip_app_os_device_day_click_time_next_2     120
- ip_os_device_count_click_id               113
- ip_var_hour                                94
- ip_day_hour_count_click_id                 91
- ip_mean_is_attributed                      74
- ip_count_click_id                          73
- ip_app_os_device_day_click_time_lag1       67
- app_mean_is_attributed                     67
- ip_nunique_os_device                       65
- ip_nunique_app                             63
- ip_nunique_os                              51
- ip_nunique_app_channel                     49
- ip_os_device_mean_is_attributed            46
- device                                     41
- app_channel_os_count_click_id              37
- ip_hour_mean_is_attributed                 21

**模型**: 除了一个lightgbm(public lb分数0.9817)之外, 都是neural nets. 构造了多个不同结构的NN来增加模型多样性: 通过RNN来对点击的时序信息进行建模; 在dense layers上加上res-link等. 因为我对各类NN还不是非常熟悉, 更多细节(数据预处理和超参数设置等)还是建议仔细阅读原贴, bestfitting也给出了部分网络结构代码, 值得一读.

**策略**: 将全部训练集分成n部分, 进行n折训练(据他的话这题的数据对他之前遇到过的数据算小了...汗颜啊, 这就是经验). 然后将n份预测得到的结果作为另一个模型的输入进行ensemble, 同时添加了一些groupby特征, ensemble对分数提升非常大.

#### (4) 第4名 (K.A.C.)

原贴: [4th place (brief) tips](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56243)

**特征工程**: 亮点特征在于对于各种ip-app-device-os的排列组合提取了[Weights of Evidence特征](https://github.com/h2oai/h2o-meetups/blob/master/2017_11_29_Feature_Engineering/Feature%20Engineering.pdf), 也尝试了target-encoding, 但是效果不好, 作者暂时未给出原因分析. 同时在[Solution to Duplicate Problem by Reverse Engineering (0.0005 Boost)](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56268)中作者也提到time_to_next_click特征在[ip-app-device-os]的分组上效果较好, 而加入channel之后效果不好, 同样的发现也可见anttip的kernel: [TalkingData Wordbatch FM_FTRL LB:0.9752](https://www.kaggle.com/anttip/talkingdata-wordbatch-fm-ftrl-lb-0-9769?scriptVersionId=3179294). 作者认为原始数据有两张表: 点击表(click_id, ip, app, device, os, channel, time)和下载表(ip, app, device, os, time), 我们见到的并非实际的下载, 下载表是根据相同的(ip, app, device, os)中最后的一次点击然后映射到点击表中. 所以channel并不重要, 因为下载表中没有channel. 

**模型**: 总共训练了50+个模型, 包括了lightgbm和neural nets. 在最后做stacking的时候, 加上了所有第一层模型用到的特征, 使得stacking分数从0.9820提升到0.9823, 作者称其为"restacking".

**训练与验证**: 采用11.07和11.08两天的数据作为训练集, 在11.09的4:00-14:00数据上进行验证预测; 最后以11.07-11.09三天的数据作为训练集再次训练来预测11.10的数据. 鉴于有重复数据, 进行了post-processing(由第2名组提出的[方案](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/55677)), 即认为几个相同的记录中最后一条记录才有可能进行下载, 前面的下载概率直接设置为0. 而该方案改进了这一方法, 使得分数提高更多, 因为click_id没有严格升序排列, 而我们是不知道哪次点击是最后一次, 作者进行了大胆猜测, 对训练集在每个(ip, app, device, os)上通过(time, is_attributed)进行排序来得到最后一次点击, 对测试集在(time, click_id)上进行排序. 最后得到了0.0005的提升, 非常强悍! 可见其深厚的数据洞察力! 细节可见[Solution to Duplicate Problem by Reverse Engineering (0.0005 Boost)](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56268)

最后, 该组已在github上开源了代码: [Code Sharing, 4th Place](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56545).

#### (5) 第5名 (MMDP)

原贴1: [5th place solution (0.9836 with pure feature engineering)](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56406)

**特征工程**: 参赛者mamasinkgs总共选择了695个特征, 细节见[特征列表](https://kaggle2.blob.core.windows.net/forum-message-attachments/326225/9385/9836_features.txt).

模型: 参赛者mamasinkgs采用lightgbm进行训练, 第一个模型基于561个特征, 第二个模型基于695个特征, 第二个模型相比第一个没有太大提升, private lb分数达到0.9836.

原贴2: [5th place story (pocket)](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56319#325540)

**特征工程**: 参赛者pocket选取了约30个特征, 较为重要的是next_clicks和ip-nunique特征. 特征不是在全部数据上抽取而是按天抽取.

**模型**: 采用lightgbm + pseudo-labeling, private lb分数达到0.9828. 本方案的创新点在于pseudo - labeling和data augmentation, 作者在之前的比赛中已有使用经验, 细节可参考原贴.

**训练与验证**: 选取5个随机种子, 最后求平均, 有0.0001的提高. 最后融合了其他队友的模型: 0.9836(mamas), 0.9827(Danijel)等, private lb分数最终为0.9840.

#### (6) 第6名 (CPMP)

原贴: [Solution #6 overview](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56283)

**特征工程**: 总共选取了48个特征, 在进行特征工程的时候, 受限于机器内存, 只能采用一定的[优化策略](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56105). 特征可以大致分为以下几类:

- 原始特征: app和os, 作为分类变量
- 时间变量: 以UTC 4PM为起始, 得到24小时制的时间特征, 用于计算lag特征
- 用户特征: (ip-device-os)看做一个用户
- 统计特征: 包括count, unique-count, time-delta (next/previous), 其中time-delta next非常重要
- 滞后(lag)特征: 通过之前一天的数值进行提取, 包括计算is_attributed的加权平均(target-encoding)
- 比率特征: 例如每个ip-app的点击与每个app的点击之比
- 基于目标变量的特征: 根据user-app-click_time进行分组, 根据is_attributed进行排序.
- 是否为最后一条记录: 0-1变量, 根据user-app-click_time进行分组
- 矩阵分解特征: 用于刻画user和app之间的相似性. 构造一个点击count的矩阵, 如ip x app, user x app, os x device x app, 这些都是系数矩阵; 对于前两个矩阵采用SVD分解(降维至3-5), 可以得到ip和user的隐变量(embeddings); 对于第三个矩阵, 采用keras的libfm进行分解. 这些特征对模型帮助很大, 有0.001的提升.

**模型**: 只采用lightgbm, 重点放在了特征工程. 最终融合了5个相似的lightgbm, 有0.0002的提升.

**训练与验证**: 采用11.09之前的数据进行训练, 在11.09的4/5/9/10/13/14时进行验证, 其中4作为public lb的验证, 两个验证分数确保不会过拟合; 接着在所有数据上进行训练, lightgbm参数(early stopping)与之前的训练过程一致(没有进行全面调参, 仅供参考: scale_positive=400, max_depth=8, num_leaves=31). 鉴于训练时间过长, 采用了以下训练和特征选择方法: 只在11.09的数据上进行训练, 抽取5%的4时数据作为验证, 所得到的验证分数和public lb分数具有一致性, 但对于lag特征效果不好; 把每个特征都存放在一个文件中, 因此每测试一个特征只需要将这个文件并入数据中, 特征通过前向选择的方式选取, 即如果验证分数提高至少0.00005就将该特征纳入模型中, 同时也采用了一点后向选择, 即先加入若干个特征, 然后一个一个移除, 查看验证分数是否下降.

#### (7) 第8名 ([ods.ai] blenders)

原贴: [8'th place solution. [ods.ai] blenders in game )](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56325)

注: 本帖只是Kruegger的参赛方案.

**特征**: 与kernel中公开的特征基本相同. 其特征选择的方式与CPMP类似.

**模型**: 采用了lightgbm, 使用贝叶斯优化来寻找最佳参数: https://github.com/fmfn/BayesianOptimization/blob/master/README.md

**训练与验证**: 用11.07的数据进行target-encoding, 用11.08和11.09的数据进行训练, 用训练集中最后的250万行数据作为验证集.

同时作者也提出了一种numpy使用技巧来进行内存控制管理, 细节可见[How to use more features to training with dark numpy magic...](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56105).


#### (8) 第9名 (Brute Force Attack)

原贴: [9th place](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56279)

第9名参赛组没有给出详细的过程, 他们使用了R里面的data.table来生成特征. 亮点在于[熵特征](https://github.com/owenzhang/kaggle-avito/blob/a7a2cc853b0ca86f07cdb9dd483779b2927b99ee/avito_utils.R). 他们通过EDA发现device id为3032基本没有下载且没有在测试集中出现, 因此直接舍去, 有0.0004的提高.

#### (9) 第11名 (tkm2261)

原贴: [11th place features](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56250)

**特征**: 使用BigQuery来进行特征抽取, 非常快.

**模型**: lightgbm, 使用不同的随机数种子进行融合. 模型参数见https://github.com/tkm2261/kaggle_talkingdata/blob/master/protos/train_lgb.py.

**训练与验证**: 使用11.09之前的数据进行训练, 11.09的数据用于验证.


#### (10) 第13名 (Alberto Danese)

原贴: [Hints on 13th place solution.. and finally Grandmaster!](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56333)

**模型**: xgboost和lightgbm (R).

**训练和验证**: 部分模型用11.07-11.09的数据, 部分模型用11.07-11.08的数据. 随机选取1000万行数据进行验证.

#### (11) 第14名 (Started From The Bottom)

第14名的方案与总结已发布在知乎: [kaggle Talking Data 广告欺诈检测竞赛 top 1%方案分享](https://zhuanlan.zhihu.com/p/36580283), 感谢@包大人.

#### (12) 第28名 (Agree with 24th trust cv)

原贴: [28th place, A 0.0006x boosting trend feature and non-overfitting target encoding (attributed rates)](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56368)

**特征工程**: 作者分享了两类特征. 第一类是target-encoding特征. 受[feature=engineering-importance-testing](https://www.kaggle.com/nanomathias/feature-engineering-importance-testing)这个kernel的启发, 中间遇到了overfitting的问题. 直观来看, 历史attributed rate对于刻画app的质量以及用户喜欢有一定帮助, 作者发现有25%的下载来自于只有一次点击的用户(ip-os-device), 还有更多用户只点击了一次但没有下载, 作者推测这可能是造成overfitting的原因. 因此他采取了一些方法来进行控制:

- 只采用之前一天的数据来生成特征, 比如11.08的特征用11.07的数据生成
- 用这一天之前的所有数据来生成特征, 可以得到更精确的attributed rate, 比上一个方法要好
- 用除了这一天的所有数据来生成特征, 这个方法得到的提升最多

作者总共提取了6个相关特征: 基于ip-app, ip-os, ip-os-device, ip-app-os-device, app-channel. 进行一定的平滑化处理也有一定帮助, 比如log(count+1), Laplace平滑等. 这边作者采用的Laplace平滑为: (downloads + m*average_rate) / (total counts + m), 其中m是可调整的参数.

第二类是趋势特征(trend feature), 这是作者在计算第一类特征时观察中间结果时发现的. 例如11.10的趋势为11.09的下载数/11.08的下载数, 总共提取了3个相关特征: 基于ip-app-os-device, ip-os-device, app. 这类特征在private lb上有0.0007的提升, 但在public lb下降了0.0001, 因此作者最后没有采用 (很可惜, 但一想如果是我很可能也会一样, 真的要对public lb持谨慎态度, 相信本地cv). 作者还提出另一种提取趋势特征的想法, 但没有时间尝试, 详细细节可见原贴.

#### (13) 第34名 (Tenuki)

原贴: [34th place, thoughts on CV and Target Encoding in R](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56304)

**特征工程**: 主要包括统计特征(groupby+count/nunique), next_click特征和target-encoding特征. 其中采用改进的target-encoding之后public lb从0.9806提升至0.9816. 作者用其最好的lightgbm模型得到训练集和测试集(test-supplement)的预测值, 然后根据预测值得到3个0-1变量: 1) 如果预测值大于0.99, 取1; 2) 如果预测值大于0.997, 取1; 3) 如果预测值大于0.999, 取1. 接着根据这3个0-1变量来进行target-encoding. 进行测试后发现0.997这个变量得到的提升最大.

**模型**: 采用了两个lightgbm, 第二个模型采用第一个模型中的非分类特征和target-encoding特征.

**训练与验证**: 在11.07和11.08上进行训练, 在11.09上进行验证, 然后采用相同设置在全部数据上进行训练. 

#### (14) 第50名 (DiveIntoTD)

原贴: [Solution #50 Story - One Week Hackathon](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56429)

**特征工程**: 主要来源于已经公开的kernel, 作者提到他们提取了"the click after next click"特征, 有一些对于模型有帮助. 除此以外, 也生成了点击率(click_rate)特征, 描绘了ip在一天内的频率. 最后总共有34个特征. 对于特征选择作者也分享了一些心得, 因为只花了10天时间, 所以采用了更快的方法, 但不清楚是否适用于别的情况, 详细细节可见原贴.

**模型**: 作者推荐了在lightgbm中设置boosting=dart, 发现dart tree在较大数据上可以比GBM表现更好. 但是dart tree的调参有一定的技巧性, 一些在GBM上的方法并不适用.

除了以上前50名的方案外, 其余还有一些参赛方案:

- [Solution #55 up on github](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56349)
- [How Boosting from 0.9671 to 0.9823 my first competition on kaggle](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56317)
- [191st Place Solution Or more like lessons learnt....](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56404)
- [560th Solution :)](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56259)

### 5. 总结

正如之前提到, 本次比赛难得的可以不用NN就取得不错的成绩, 我也是第一次参加这样纯数据挖掘的比赛, 在对于较大数据集的处理和特征工程方面有很多的收获. 可以看到, 优胜方案中有通过仔细分析业务背景得到创新特征的, 也有通过开挂一般的数据洞察力进行结果修正的, 这些都让我打开眼界, 也是本次比赛我觉得最重要的内容.

最后一天我还在铜牌区(250-300左右), 但由于有人提早公开了一个public lb可达到0.9811的提交结果, 加上"热心"网友在最后半天公开0.9812的blend of blend kernel, 导致最后跌落到600名, 实在有点可惜, 与第二枚kaggle奖牌失之交臂. 但总的来看, 这次学习得到收获的喜悦要大于结果不佳的失落, 尤其是赛后围观前排大佬的方案真是让人惊艳. 希望今后比赛能吸取教训, 继续努力, 学到更多干货.