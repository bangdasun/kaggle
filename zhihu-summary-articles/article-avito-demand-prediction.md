## Kaggle竞赛-Avito Demand Prediction Challenge小结

竞赛网址: https://www.kaggle.com/c/avito-demand-prediction

### 1. Avito

Avito总部位于莫斯科, 是俄罗斯最大的分类信息网站, 一些具体信息可以参考[紧跟 Craigslist 和58同城，Avito 跃居为第三大分类网站](http://tech2ipo.com/58384). 本次比赛针对的是Avito网站上二手商品买卖业务.

### 2. 比赛简介

#### (1) 业务背景

比赛主题写着Demand Prediction, 但实际上就是预测二手商品的成交概率(deal probability). 


#### (2) 数据
Avito提供了多个数据集, 除了主要的训练集(train.csv, 150万+样本)和测试集(test.csv, 50万+样本)外, 还有train_active.csv/test_active.csv(和train/test同时期的数据, 但没有成交概率和图片信息); periods_train.csv/periods_test.csv(只包含了active数据中每件商品的id, activation_date - 广告投放日期, date_from/date_to - 广告展示的第一天和最后一天); train_jpg.zip/test_jpg.zip(train和test样本的图片).

train和test中每条记录是一件商品广告, 包含的字段为:

- item_id: 广告id
- user_id: 发布广告的用户id
- region: 用户所在地区
- city: 用户所在城市
- parent_category_name: Avito广告模型对商品的一级分类
- category_name: 更细致的商品分类
- param_1: 可选参数1
- param_2: 可选参数2
- param_3: 可选参数3
- title: 广告标题
- description: 广告描述
- price: 商品价格
- item_seq_number: 广告序列号
- activation_date: 广告投放日期
- user_type: 用户类型
- image: 图片id
- image_top_1: Avito对于图片的分类代码
- deal_probability: 成交概率(目标变量)

这里的param_1, param_2, param_3类似于关键词/补充细节描述, 比如商品的品牌, 尺寸等, 有不少是缺失值.

#### (3) 结果评判标准

比赛最终要求提交每件商品的成交概率, 以此计算得到的RMSE(Root Mean Squared Error)作为评判标准.

### 3. 我的参赛方案

半年前kaggle有个类似的比赛[Mercari Price Suggestion Challenge](https://www.kaggle.com/c/mercari-price-suggestion-challenge), 关于通过二手商品类别/描述等特征来预测价格. 做过这个比赛的老铁们应该在一开始就意识到有不少方(tao)法(lu)是可以用到Avito中的. 同时因为有不少分类变量, 所以之前TalkingData比赛也有可以很多借鉴, 发现我一个月之前写的TalkingData竞赛总结还是非常有帮助的;)

下面就简单介绍下我的参赛方案(95/1917).

#### (1) 特征工程:

- 原始特征:

region/city/parent_category_name/category_name/param_1/param_2/param_3/image_top_1都属于分类变量, 可以用LabelEncoder转化后输入lightgbm, 也可以用OneHotEncoder转化后输入Ridge和各类NN. 训练集和测试集的user_id重合度很低, 所以user_id没有列入特征. 对于缺失的image_top_1进行了填补, 采用parent_category/category分组中最多的一类作为填补值;

price/item_seq_number, 属于数值变量; 

activation_date, 可以提取出day_of_week, day_of_month特征;

title/description, 可以用bag of words/Tfidf转化得到稀疏特征. 基于Mercari的经验, 我把parent_category_name/category_name/param_1/param_2/param_3加入到了title和description之中.

- 统计特征: 

和TalkingData类似, 可以通过不同的分类变量组合计算count, nunique, entropy等统计量(比如每个用户发布的广告数量, 用户发布广告的频率等), 计算price的mean/median/max/min/std统计量, 以及用原始价格减去这些统计量得到价差特征;

可以从title和description中提取一些文本统计特征, 比如文本长度/文本单词数/文本句子数/大写字母数/标点数, 也可以更进一步计算一些比率特征, 比如标题长度与文本长度的比率, 文本单词数和文本长度的比率等. 还计算了title和description的edit distance作为相似度特征.

- 嵌入/分解特征:

这部分特征源于TalkingData比赛第一名的方案, 通过一个分类变量进行分组, 然后得到另一个分类变量的序列, 将其看做文本, 接着用CountTokenizer进行转换, 最后用LDA/NMF/LSA进行矩阵分解/降维处理得到数值变量. 我提取了user_id和image_top_1, param_1, category_name等变量的嵌入特征, 提取LDA和NMF转化后的前5列作为特征.

受这个特征的启发, 我也对title/description的Tfidf矩阵进行LDA/NMF降维, 提取了前50列作为特征. 每个title和description都对应着一个50维的向量, 所以我还计算了两个向量的cosine/Manhattan/Euclidean距离作为相似度特征.

- 图像特征:

这部分特征完全参考kernel: [Ideas for Image Features and Image Quality](https://www.kaggle.com/shivamb/ideas-for-image-features-and-image-quality), 但很奇怪的是加入这部分特征后public lb分数下降不少, 连最基本的图像宽度/大小等特征也不行(讨论区有好几人提到这些特征是有帮助的), 所以最后我删去了全部的图像特征, 赛后发现private lb也是下降的. 

- 弱模型特征:

这部分特征思路源于kernel: [LightGBM with Ridge Feature](https://www.kaggle.com/demery/lightgbm-with-ridge-feature), 即采用较弱模型得到out-of-fold预测值作为训练集特征, 预测值作为测试集特征, 这其实已经属于模型Stacking的思路了. kernel中是用title和description的Tfidf稀疏特征训练了一个Ridge回归, 我觉得如果加入更多类似特征应该会有帮助, 所以我还采用了FM_FTRL/SVR/基于self-trained word2vec的RNN/基于HashingVectorizer的Ridge回归等. 加入这些特征之后确实有一定提升, 但随着特征增加提升是越来越小的, 可能是因为模型之间差异性不大(计算相关系数可以发现部分模型的相关性有0.98左右).

#### (2) 模型:

- LightGBM: 最主要的模型, 我最后提交的也是加入了弱模型特征的lightgbm预测结果, RMSE为0.2189;
- FM_FTRL/Ridge/SVR/self-trained w2v + RNN;
- MLP: 参考kernel - [Boosting MLP](https://www.kaggle.com/peterhurford/boosting-mlp-lb-0-2297), RMSE为0.2262 最后发现对总体分数没有提升, 所以未使用;
- RNN/CNN: 根据kernel - [RNN_Detailed Explanation_0.2246](https://www.kaggle.com/shanth84/rnn-detailed-explanation-0-2246)进行了一定改进, 将RMSE降低至0.2241, 但后来时间不够且发现lightgbm那边还可以挖掘不少特征, 所以最后也没有使用.

#### (3) 训练与验证:

虽然这次比赛的数据是带时间戳的, 且public lb和private lb也是根据时间划分的(参考kernel - [Test on LB split](https://www.kaggle.com/cczaixian/test-on-lb-split)), 但我还是采用随机划分的方式划分了训练集(90%)和验证集(10%), 因为数据在日期上的分布不均匀, 最后几天的数据非常少, 而增加时间窗长度后训练集又太小. 后面所有的模型都用这套验证方式(保证随机数种子都一致).

### 4. Top参赛方案

接下来又是本文重点了, 这里声明因本人对于NN的使用经验较少, 一些翻译和解读可能不够准确, 如有疑惑请参考原贴;)

#### (1) 第1名 (Dance with Ensemble)

原贴: ["Dance with Ensemble" Sharing Thread](https://www.kaggle.com/c/avito-demand-prediction/discussion/59880)

第1名团队每位成员都分享了自己的工作, 他们搭建了3层模型融合(Stacking).

Little Boat: 主要分享了他所构建的NN模型细节.

特征包括了文本特征, 分类特征, 数值特征, 图像特征; 构建步骤大致如下
 - 数值特征 + 分类嵌入特征, 分数为0.227X; 
 - 通过两个RNN加入title和description, 使用FastText嵌入, 调参之后分数为0.221X; 
 - 通过self-trained FastText嵌入(train+test), 分数为0.220X;
 - 加入VGG16的Top layer with average pooling, 在融合文本/分类/图像/数值特征之前加入了额外的一层, 通过一定的调参之后达到0.219X;
 - 尝试文本模型, CNN + Attention, 效果不好; 最后使用了两层LSTM+Dense layer, 大约有0.0003的提升;
 - 尝试讲图像输入不同的CNN, 最后采用了ResNet50的中间层, 大约有0.0005的提升;
 - 调参中发现正在Text和LSTM之间加入Spatial Dropout有一定帮助(0.0007-0.001), 最后总的Dropout也有一定帮助. 最后提升了0.001-0.0015, 此时模型分数在0.2165-0.217左右;
 - 加入了队友构造的特征, 达到了0.215X;
 - 最后把所有的模型放到一起, 再训练一个全连接的NN, 有0.0008的提升.

最后Little Boat展示了其NN结构:

![Little Boat's NN](https://pbs.twimg.com/media/DgvX3pWUYAAlCAK.jpg:large)

Arsenal: 主要分享的是lightgbm和xgboost的模型细节.

**特征工程**: 

- 文本特征. 对title/description/title+description/title+description+param_1等进行Tfidf处理, 对于xgboost是直接输入, 对于lightgbm是进行SVD降维后再输入, 以保持模型的多样性; 文本统计特征, 包括文本长度, 文本长度与词数的比率等;
- 图像特征. 图像的基本特征(与我参考的图像特征kernel一样); 从pre-trained NN(ResNet50, InceptionV3, Xception)提取的特征, 取了top-5的种类作为新的分类变量(类似于商品种类); VGG16特征, 512个原始特征+15个PCA降维后的特征; [key-point特征](https://www.kaggle.com/c/avito-demand-prediction/discussion/59414);
- 分类特征. 对不同的分类变量组合提取count/nunique特征, 是在train+test和train+test+train_active+test_active上提取的; Target-encoding, 提取了频数大于5000的类别的mean deal_probability, OOF mean deal_probability, OOF mean deal_probability x min{1, log(count)/log(10000)};
- 预测特征. 对price, image_top_1, item_seq_number, day_diff(day_to - day_from)分别构建了预测模型. 也通过不同的分类变量组合提取mean price, image_top_1, item_seq_number, day_diff特征, 同时也进行了做差得到例如price_diff的特征;
- user_id特征. 这里细节较多, 建议参考原贴, 囧..

对于特征选择, 作者选取了能获得模型gain 99%的特征, 特征数分别为700+(第一层), 300+(第二层).

**模型/训练与验证**: 不同层的lightgbm采用了3种不同的参数设置, 通过bayesian optimization来搜寻最优参数. 作者也尝试将训练集中不同level的deal_probability作为一个类别来训练分类模型.

thousandvoices: 构建了多个lightgbm/xgboost/catboost/NN.

**特征工程**:

- 分类交互特征. 例如region+parent_category_name等等;
- 分类编码特征. 对于每个分类变量都加入了以下特征: 之前广告的mean deal_probability, count, median price;
- 文本特征. 与之前Little Boat/Arsenal提到的基本一致;
- 统计特征. min/median/max price, 根据title分组得到count特征等;
- 预测特征. 使用train_active/test_active数据训练了可以预测price和length of period for ad的NN模型, 这些都是强特征.

**模型/训练与验证**: 采用了深层的boosting tree, 其中lightgbm的num_leaves为1024.


Georgiy Danshchin:

**特征工程**:

- 分类特征. 例如对category/user_id的Percent encoding, mean target encoding, 价差特征等.没有采用分类变量的交互特征. 对分类变量进行one-hot处理后输入lightgbm;
- 文本特征. 对title+description+param抽取bigram的Tfidf, 文本统计特征, 14个Ridge prediction特征等;
- 数值特征. 对price和item_seq_number进行log1p变换;
- 图像特征. channel mean, 图片宽度和高度, ResNet152的最后一层输出及SVD降维后的前100个特征等;

这里的亮点在于对于active数据的处理, 他在active数据上训练了3个NN模型来预测log1p(price), renewed和log1p(total_duration), 这些预测值与log1p(price)的差值是很重要的特征.

**模型/训练与验证**: 在与队友组队前采用4折移动时间窗验证(细节见原贴). 线下验证分数与public lb分数基本变动一致. 总共训练了25个模型(15个lightgbm和10个NN), 在不同的特征子集上进行训练. 总结一下就是构造模型之间的多样性以便于最后的模型融合.

#### (2) 第2名 (Song and Dance Ensemble)

原贴: [second place solution](https://www.kaggle.com/c/avito-demand-prediction/discussion/59871)

**特征工程**:

- 图像特征: 采用pre-trained VGG16/ImageNet/ResNet50/MobileNet提取特征;
- 文本特征: 在全体数据上训练了FastText, 接着生成了title/description/title-city/title-category/stemmed title/stemmed description对应的向量;
- 统计特征: 对于每个分类特征/二级交互/三级交互组合都计算了mean price, 计算了每个广告的mean number of days active等
- 非监督学习: 对于分类特征训练了AutoEncoder, 称为user2vec模型, 细节可见原贴.

**模型/训练与验证**: 主要是NN和lightgbm. 

- NN: 最好的NN模型有多个分支: 分类特征的嵌入, 数值特征, FastText向量特征, 图像向量特征, 文本的BiLSTM, target encoding特征. 具体细节见原贴.
- LightGBM: FastText训练得到的向量特征很有帮助, 对Tfidf进行SVD分解也有帮助.
- FM_FTRL/Ridge/Catboost

最后搭建了6层模型融合. 验证方式为简单的10折交叉验证.

#### (2) 第3名 (SuperAnova)

原贴: [3 place solution](https://www.kaggle.com/c/avito-demand-prediction/discussion/59885)

**特征工程**(LightGBM): 

- 文本特征: 在description上的bigram特征, 在title+param上的unigram特征, char-level的5-gram特征, Word2Vec得到的向量特征, pre-trained FastText得到的向量特征;
- 图像特征: VGG16/VGG19/ResNet/Inception/Xception得到的top-3分类特征;
- 数值特征: 对price进行分箱处理, 一些统计特征和地理位置特征;
- 分类交互特征: 对三级交互分类变量组合提取likelihood和count特征.

**模型/训练与验证**: 主要是NN和lightgbm, 其中lightgbm采用xentropy作为objective, 而不是regression.

**NN模型**: 最好的NN是两个结合FastText和Word2Vec以及其他特征的Stacked bidirectional GRU模型. 所有的分类变量都采用了长度为100的嵌入, 训练目标函数为binary cross-entropy, 输出层的激活函数为sigmoid函数. 其余细节见原贴.

#### (4) 第4名 (wave in the distance at the top)

原贴: [4th Place Solution](https://www.kaggle.com/c/avito-demand-prediction/discussion/59881)

**特征工程**: 除了Tfidf稀疏特征外总共有800个特征, 亮点在基于价格做了不少工作.

- 文本特征: Tfidf稀疏特征有100000个(title+description+param_1), 采用的是trigram.
- 价格特征: 作者认为价格是非常重要的特征, 所以尽可能地刻画了价格的分布以及信息. 对于每个分类组合, 他提取了20th percentile/median/max/std/skew, 然后计算其与实际价格的比值. 

作者的一个重要发现是把价格统计特征应用到一些text groupings上非常好: title_noun aggregates - 从title中提取名词, normlize之后按字母顺序排序, 然后作为关键词来作为分组依据; title_noun_adjs aggregates, 提取形容词; title_cluster 对title的Tfidf进行SVD降维(500维), 然后做一个k=30000的Kmeans聚类(mini-batch训练); text_cluster, 同样的方法也用到title+description+param上. 这个思路是作者从用户使用的角度上出发的, 用户会根据关键词来进行搜索, 然后按价格排序来浏览商品.

作者的另一个发现叫做User semantics, 即提取用户信息. 对train/test中的用户, 对title+description+params进行HashingVectorize处理而不是普通的Tfidf, 然后用SVD降维到300作为特征.

- 图像特征: 提取了blurrness, color histograms (包括SVD降维特征和一些统计特征), NIMA等. 

除了上述特征之外, 还提取了一些kernel中被用到的特征, 比如地理位置的聚类类别等等.

**模型/训练与验证**: lightgbm参数 - 采用较深的深度和较小的学习率.

除了这些之外, 作者Joe也分享了不少对比赛有帮助的干货, 细节可以参考他的原贴;)

#### (5) 第5名 (Optumize)

原贴: [5th place solution](https://www.kaggle.com/c/avito-demand-prediction/discussion/59914)

**特征工程**: 较强的特征是与价格有关的, 作者采用了Bayesian mean与group mean的比值: ((item_price/mean_price_grp)*ct_grp + (prior))/(ct_grp+prior), 使用了group count进行加权: 2件商品中便宜的和100件商品中最便宜的有不小的差别. 通过这一步有0.002的提升. 代码实例可见其[github](https://github.com/darraghdog/avito-demand/blob/master/features/code/pratioFestivitiesR1206.R).

**模型**: 采用了MLP/RNN/LightGBM/Ridge模型, 其中部分模型的代码已分享至作者的[github](https://github.com/darraghdog/avito-demand).

#### (6) 第7名 (Light in June)

原贴: [7th place solution](https://www.kaggle.com/c/avito-demand-prediction/discussion/60026)

**特征工程**: 大部分特征比较常规, 与其他队的类似. 不同之处在于作者分享了一些失败的特征.

**模型**: 基于Bayesian寻优的lightgbm和xgboost/NN/Ridge, Drop0 model. 一些具体的细节较为零散, 且相应作者还在准备当中 [待更新].

#### (7) 第13名 (Win a gold for my grandma)

原贴: [The last gold solution](https://www.kaggle.com/c/avito-demand-prediction/discussion/59886)

**特征工程**: 除了来自于kernel和讨论区的特征之外, 对title在category_name上的分布进行聚类得到title_cluster, 伪代码可见原贴. 接着基于title_cluster又做了一些计算, 比如count, mean price等. 作者还用正则表达式从title中提取出一些数值信息, 比如房间面积, 使用年份等, 来计算一些相关的价格特征.

**模型**: 作者使用的NN由BiLSTM on text+4 layers CNN on image组成, 以此生成图像的向量表示来作为特征, 但发现用相同的方法来处理文本时造成了过拟合, 因此他只采用title和description的Word2Vec来预测price, item_seq_number, city等一系列分类特征. 

#### (8) 第14名 (The Almost Golden Defenders)

原贴: [14th Place Solution: The Almost Golden Defenders](https://www.kaggle.com/c/avito-demand-prediction/discussion/60059)

本帖作者对于模型融合进行详细描述, 而关于特征工程方面他表示下周会更新 [待更新].


#### (9) 第18名 (we had great fun)

原贴: [The way to the 18th place, just for rookies.](https://www.kaggle.com/c/avito-demand-prediction/discussion/59922)

**特征工程**: 除了kernel上一些公开的特征以外, 还做了keywords count, text hash, categorical feature hash等. 对于价格, 采用指数和线性分段处理, 作为新的分类特征. 还加入了一些外部数据, 如城市地区人口, 收入等.


#### (10) 第25名 (addimewe)

原贴: [25th place solution](https://www.kaggle.com/c/avito-demand-prediction/discussion/59902)

**特征工程**: 有两个有趣的发现 - 1) 每个用户的重复数, 在kernel [Aggregated features & LightGBM](https://www.kaggle.com/bminixhofer/aggregated-features-lightgbm)里重复值是被移除的, 而作者发现这些重复值也可以作为有效的特征; 2) 对于同一用户, 根据item_seq_number可以得到广告的投放顺序, 基于此计算price difference有很大的帮助. 除此之外一些外部数据比如城市人口特征也有一定帮助.

**模型**: 作者给出了NN模型的代码, 细节可见原贴.

其他也还有一些队伍分享了方法:

- [20th place solution](https://www.kaggle.com/c/avito-demand-prediction/discussion/59936)
- [22nd place solution Team NoVices](https://www.kaggle.com/c/avito-demand-prediction/discussion/60102)
- [Our 30th Solution: In which our heroes tried Quantum Gravity, Adaptive Noise and other cool stuff…](https://www.kaggle.com/c/avito-demand-prediction/discussion/60006)
- [A silver solution (31st place)](https://www.kaggle.com/c/avito-demand-prediction/discussion/59959)
- [36th place: squeezing out the last -0.0011 RMSE](https://www.kaggle.com/c/avito-demand-prediction/discussion/59872)
- [37th place solution](https://www.kaggle.com/c/avito-demand-prediction/discussion/59899)
- [Place 40, tried to get something from train_active and test_active with DAE](https://www.kaggle.com/c/avito-demand-prediction/discussion/60005)

最后也列一些我参考过的kernel:

- [Self-trained embeddings Starter (only description)](https://www.kaggle.com/christofhenkel/self-trained-embeddings-starter-only-description)
- [Using train_active for training word embeddings](https://www.kaggle.com/christofhenkel/using-train-active-for-training-word-embeddings)
- [Aggregated features & LightGBM](https://www.kaggle.com/bminixhofer/aggregated-features-lightgbm)
- [Modified Wordbatch + Ridge + FM_FTRL + LGB](https://www.kaggle.com/peterhurford/modified-wordbatch-ridge-fm-ftrl-lgb)
- [Ideas for Image Features and Image Quality](https://www.kaggle.com/shivamb/ideas-for-image-features-and-image-quality)
- [Stacked Model (CNN+XGBoost)](https://www.kaggle.com/jingqliu/stacked-model-cnn-xgboost)

### 5. 总结

这次比赛难得的集中了tabular data/text data/image data, 算是综合性很强的比赛了, 比较贴合实际. 所能挖掘的地方也很多, 学到了不少东西. 总的来看, 优胜方案都充分利用了数据, 即train_active/period_train这些数据, 且很好地使用了NN模型. 对于这样的比赛, 模型融合是很重要的, 这也是我本文没有过多涉及的地方, 如果有兴趣的话建议参考原贴.

作为新手, 我觉得比赛的时候是很容易被kernel区的开源所影响的, 甚至被牵着鼻子走: kernel区没有好的开源自己也没法提升分数, 这是我第一次比赛(Mercari)时候的感受. 想要有所收获, 还是要在比赛之外学习不少知识, 这样才能保证有自己的思路.

最后, 我对比赛成绩(95/1917)也比较满意, 获得银牌区最后一名, 运气不错. 回想一个月前TalkingData那次遗憾的经历, 这次比赛终于有了好运. 接下来也期待kaggle能够举办更多有意思的比赛.