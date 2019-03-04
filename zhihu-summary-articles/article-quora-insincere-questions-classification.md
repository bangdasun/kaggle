## Kaggle竞赛-Quora Insincere Question小结

竞赛网址: https://www.kaggle.com/c/quora-insincere-questions-classification

比赛已经结束快3周了, 因为一直还在做Elo Merchant Category Recommendation, 所以赛后总结到今天才有空写, 看了下知乎上已经有不少了, 摘取了几个总结比较全面的(如果有遗漏欢迎补充): 
- [Quora Insincere Questions Classification赛后的一些思考](https://zhuanlan.zhihu.com/p/56435041)
- [Quora Insincere Questions Classification比赛Salon sai](https://zhuanlan.zhihu.com/p/57550039)
- [kaggle quora轻松进5%](https://zhuanlan.zhihu.com/p/54741029)
- [Kaggle混分记](https://zhuanlan.zhihu.com/p/56747391)

这次和几个大佬组队, 最终排名76, private分最高的也是local CV最高的, 算是没有遗憾. 赛后的重点还是总结思考, 看看前排大佬是怎么做的, 自己动手实践, 这样才有进步提高.

### 1. Quora

相当于知乎美国版, 相信大家都比较熟悉了.

### 2. 比赛简介

#### (1) 背景

Quora希望能通过模型来自动识别用户提出的问题是否是insincere的, 可以理解说用户是不是真心的在说人话/问问题而不是在带节奏或者问一些没人看得懂的问题, 很显然是文本分类问题. Insincere的具体定义可以看Quora在比赛页面的定义.

#### (2) 数据

数据的结构非常简单, 训练数据总共3列: 问题ID, 问题内容(question_text)和标签(0 - not insincere, 1 - insincere). 训练数据大约为131万, 第一阶段测试数据为5万6, 第二阶段为32万. 主办方提供了4种预训练好的词向量: GoogleNews, GloVe, Paragram, FastText.

#### (3) 结果评判标准

以kernel的形式提交. 比赛分为两个阶段, 第一阶段的测试数据占15%, 然后kaggle会在第一阶段结束后重新运行参赛者提交的kernel, 预测得到剩下85%数据的预测值作为最终提交. 如果kernel开GPU则限制时长2小时, 不开GPU则为6小时. 所以首先要保证kernel能在规定时间完成模型的训练和测试, 这样才能取得有效的成绩. 最终预测的是每个问题ID是否是insincere, 以F1-score为评价指标.

### 3. 我的参赛方案

这里首先总结下我在组队之前的方案. 这次比赛和2018年3月结束的Toxic Comment Classification类型相似, 因此开始的时候许多思路都是可以直接借鉴的, 尤其是Toxic的第3名在赛后公开了最好的单模, 所以现成的模型结构起点是比较高的.

#### (1) 预处理

这里的预处理主要指数据清洗, 比如特殊符号的处理, 拼写错误纠正等. kernel区有人分享了一种对特殊字符的处理方法, 左右各加一个空格以将其分离出原句. 从local F1 CV和public LB来看都是有所提升的. 但为什么这么做以及为何能提升, 我暂时还没有深入思考过. 其他我进行的预处理包括拼写纠正, 比如英语骂脏话时常常用*来代替一些字母, 这在Toxic比赛中以及有一些收集了, 所以这次可以直接用. 其他有人提出的方法比如替换LaTeX字符, 数字替换等, 都降低了local F1 CV, 所以没有采用. 对于文本预处理, [@程序](https://www.zhihu.com/people/cheng-xu-70/activities)老师在他的总结文中有进行思考: [Quora Insincere Questions Classification赛后的一些思考](https://zhuanlan.zhihu.com/p/56435041).

#### (2) 模型

我尝试了比较多的模型结构, 较好的结构是 `Embeddings(0.5 GloVe + 0.5 Paragram) -> Dropout2d -> BiLSTM -> BiGRU -> Atten Layer -> (skip connection) MaxPool + AvgPool -> Concat -> Dense -> Dropout -> Dense(1)`. 我尝试了:

- 把第二层BiGRU改成BiLSTM
- 把BiGRU最后的状态(last states)加入Concat
- 在BiGRU层后加卷积层(Conv1D)
- 把多个Word Embedding拼接(concat)到一起
- 移除Attention层
- 使用Capsule Net
- 在Concat层和Dense层之间加BatchNorm
- 加入更多的Dense层
- 加入文本统计特征

local F1 CV都没有超过上面结构.

输入序列的max_length为70, max_features为200000, 训练的batch_size主要为256或512, epoch为4. 采用Adam优化器, 学习率为0.001. 采用Cyclic LR可以有效提升local F1 CV. 训练采用5折交叉验证, 首先生成概率预测值, 然后根据验证集上的表现确定概率阈值将概率转化为0-1的标签. 最终采用Cyclic LR的local F1 CV为0.6915, 不采用Cyclic LR的local F1 CV为0.6850.

比赛开始阶段主要采用keras, 后来大家都发现keras在开GPU的时候因为随机性不能复现结果, 而采用Pytorch之后随机种子可以进行设定, 所以后来我用的是Pytorch.

组队后团队的方案在之后允许的情况下会进行更新 ;)

### 4. Top参赛方案

#### (1) 第1名 (The Zoo)

原贴: [1st place solution](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80568)

**模型**: 模型结构比较简单, 只采用BiLSTM + Conv1D + MaxPool + Statistical Features.

**词向量处理**: 没有将文本进行小写化处理; 尽量提供预训练词向量对实际文本的覆盖率, 比如修正单复数形式, 移除特殊字符等. 对于OOV的词采用随机向量(和public kernel的处理一样). 

**概率阈值**: 和根据验证集确定阈值不同, 作者通过多次模型训练找到一个固定的概率阈值: 对预测的概率值进行排序的平均(概率越高序号越大)而不是直接的平均, 然后计算每折最佳的F1和固定阈值确定F1的偏差, 使用产生偏差最小的阈值作为最终的阈值. 这省去了划分验证集来确定阈值的步骤.

**优化训练时间**: 采用batch level的动态sequences padding, 即不同batch的pad/truncate后长度不一样, 不是统一设定的max_length, 而最佳长度由batch中所有长度的95%分位数确定. 这在不损失预测精度的情况下大幅优化了运行时间, 使得可以融合更多模型(最终融合了10+个模型).

**模型训练**: 采用Nadam + Cyclic LR. batch_size为512.

**模型验证**:  (感觉不太好翻译和理解, 作者认为许多人的交叉验证是存在问题的, 他解释了他们的验证方法, 这里我暂时跳过).

作者也分享了一些失败的尝试, 比如shanpshot learning, pseudo labeling, 拼写纠正等. 细节可见原贴.

#### (2) 第2名 (takapt)

原贴: [2nd place solution](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/81137)

**预处理**: 在除了字母和数字之外的字符两边插入空格, 然后使用keras tokenizer(按空格划分). 然后对OOV词进行拼写纠正, 主要是通过计算编辑距离(levenshtein edit distance). 但是这些并没有使local CV有太大变化.

**模型**: 模型结构比较简单, 只有一层BiGRU和统计特征(词语数, 去除重复后的词语数, 字符数, 大写字符数, bag of characters) + Dense层. 

**词向量处理**: 拼接了GloVe和FastText, 同时也根据比赛数据训练了64维的词向量, 此外还加入4个0-1特征(是否全是大写字母, 第一个字母是否是大写, 是否只有第一个字母是大写, 是否是OOV词), 所以最后词向量的维度为300+300+64+4=668.

**训练**: 和第1名的方案一样也使用动态sequences padding, 得以训练出6个模型(不同随机种子). 使用Adam优化器, 学习率为0.001, 每个epoch后衰减0.8(即第二个epoch学习率为0.0008). batch_size调整为320, 每个模型训练5个epoch. 没有采用交叉验证训练, 所有模型都在全部数据上训练, 交叉验证在本地完成.

#### (3) 第3名 (Guanshuo Xu)

原贴: [3rd place kernel](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80495)

作者进行了大量的预处理工作, 采用了两层Global MaxPooling和checkpoint ensemble. 作者直接将方案发布在了kernel区.

#### (4) 第4名 (KF)

原贴: [4th place solution (with github)](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/81632)

**整体结构**:

![w2v-finetune](https://raw.githubusercontent.com/k-fujikawa/Kaggle-Quora-Insincere-Questions-Classification/master/overview.png)

**预处理**: 和public kernel的处理基本一致, 同时将keras Tokenizer中默认的过滤标点去除.

**词向量处理**: 为了提升预训练词向量对quora数据集的覆盖率, 对词向量进行了fine-tuned处理(CBOW模型). 为了可以融合处理前的词向量和处理后的词向量, 将词向量降维至400(随机), 在减少计算复杂度的同时也提高了模型多样性.

**统计特征**: 是否是OOV词; IDF(inverse-document-freq); 字符数量; 大写字符数量; 大写字符占比; 词语数量; 非重复词语数量; 非重复词语比例. 

作者已在kernel区和github开源: <https://github.com/k-fujikawa/Kaggle-Quora-Insincere-Questions-Classification>.

#### (5) 第7名 (yufuin)

原贴: [7th place solution - bucketing](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80561)

**预处理**: 采用nltk包进行一通猛如虎的操作, 不过作者最后说并没有带来大的收益, 只是因为花时间太多了才进行分享, 2333.

**模型**: 和其他分享的模型结构大同小异, 拼接了第一层BiLSTM的AvgPool, 第二层BiLSTM的MaxPool和Attention输出, Dense层采用tanh激活函数. 最后采用checkpoint ensemble, 对CV有0.006的提升. 

**模型训练**: 在sequences padding中使用bucketing, 和第1/2名的动态sequences padding原理类似.

### (6) 第10名 (tks)

原贴: [10th place solution - Meta embedding, EMA, Ensemble](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80718)

**预处理**: 只对标点字符进行处理, 拼写纠正没有帮助.

**模型**: 在Embedding层和BiGRU层之间加入了Dense层(ReLU激活函数), 称为**Projection Meta Embedding**. 最后融合了6个相同结构的模型. 

作者已将最后的提交发布在kernel区.

#### (7) 第15名 (Great patience)

原贴: [15th Solution - focus on models](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80540)

**模型**: 融合了RCNN和3个结构相似的LSTM模型, 使用GloVe和FastText取平均的embedding, max_length为57, max_features为tokenization后全部词语数.

作者已将最后的提交发布在kernel区.

#### (8) 第18名 (Toguro)

原贴: [18th place solution from 300-th at Public LB](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80696)

**模型/训练**: 采用2层RNN模型(宽度96)进行5折交叉验证; 对embedding层使用semantic bernoulli dropout; 加大后面epoch的batch_size; 在最后一个epoch对embedding的权重进行更新; 选到了比较好的随机种子lol.

后面作者也分享了一些失败的尝试, 细节可见原贴.

#### (9) 第20名 (I have no idea)

原贴: [20th solution - 2 models, various embeds, mixed loss](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80527)

**模型**: (1) 拼接GloVe和FastText + LSTM + CNN(kernel size = [1, 2, 3, 4]) + 2 Dense layers, 加入了BatchNorm和Dropout; (2) 平均GloVe和FastText + LSTM + GRU + (MaxPool, AvgPool) + 2 Dense layers, 加入了Dropout layers. 

**模型训练**:  采用4折交叉验证. 损失函数使用BCE + soft F1 loss, 对public LB有0.003的提升, 对local CV有0.002的提升; 使用cosine cyclic LR, max_lr = 0.003. 每个模型训练4个epoch; AdamW(weight_decay = 0.0001)可以取得更好的结果, 但比较耗时.

作者已将方案发布在kernel区.

#### (10) 第22名 (SuperTeam bit.ly/2QBpVNv)

原贴: [22nd Solution - 6 Models and POS Tagging](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80514)

**模型**: 总共6个模型(总计训练了74个epoch), 重点在于首先训练了一个模型可以快速过滤容易分类的样本(占70%), 然后在剩下30%的数据上训练其他模型: (1) DPCNN; (2) BiGRU + LSTM (GloVe embedding); (3) Parallel LSTM + GRU (w/GloVe embeddings); (4) POS BiLSTM + GRU (w/Paragram embeddings); (5) POS Parallel LSTM + GRU (w/News embeddings). POS是指part of speech, 在ensemble的时候可以提升分数.

#### (11) 第25名 (danzell)

原贴: [25th place solution - unfreeze and tune embeddings!](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80542)

作者使用的是R的keras包.

**词向量预处理**: 只选取在训练数据中的词语, OOV(GloVe/Paragram)词向量用全0向量代替. 在模型训练的最后一个epoch设置embedding为可训练.

**模型/训练**: 采用6折训练, 模型结构可见原贴; 每折训练4个epoch, 学习率分别为0.003, 0.003, 0.003, 0.001; batch_size分别为512x2(前3个epoch), 512x1.5(最后一个epoch), max_length设置为60. 最后一个epoch将embedding设置为可训练使得OOV词可以在前3轮epoch后进行一定的学习.

除此之外其他还有一些参赛方案的分享:

- [13th place solution](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80499)
- [From 400-ish public to 26 private](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80544)
- [27th kernel](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80494)
- [33rd place solution- FastText embedding](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80577)
- [38th solution, data driven to find embedding weights](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/81697)
- [44th solution: Add all the randomness or how to improve your ensemble when all your models suck](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80589)
- [70th position. Pytorchtext model](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80507)
- [117th solution. Achieve 0.701 PB and 0.703 LB in 4000 seconds.](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80509)
- [125th solution. 12folds average](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80664)

Kernel区也有一些队伍直接发布了他们的kernel.

最后列一些我参考过和认为不错的kernel:

- [Deterministic neural networks using PyTorch](https://www.kaggle.com/bminixhofer/deterministic-neural-networks-using-pytorch)
- [LaTeX cannot be used in a Quora question?!](https://www.kaggle.com/sunnymarkliu/latex-cannot-be-used-in-a-quora-question)
- [BiLSTM-attention-Kfold-CLR-Extra Features-capsule](https://www.kaggle.com/spirosrap/bilstm-attention-kfold-clr-extra-features-capsule)
- [Temporal Convolutional Network](https://www.kaggle.com/christofhenkel/temporal-convolutional-network)
- [Single RNN with 5 folds - snapshot ensemble](https://www.kaggle.com/shujian/single-rnn-with-5-folds-snapshot-ensemble)
- [Single RNN model with meta features](https://www.kaggle.com/shujian/single-rnn-model-with-meta-features)
- [Blend of LSTM and CNN with 4 embeddings (1200d)](https://www.kaggle.com/shujian/blend-of-lstm-and-cnn-with-4-embeddings-1200d)
- [2DCNN textClassifier](https://www.kaggle.com/yekenot/2dcnn-textclassifier)
- [How to: Preprocessing when using embeddings](https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings)
- [Improve your Score with some Text Preprocessing](https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing)

### 5. 总结

这次比赛与一年前Toxic Comment Classification比赛的任务目标是类似的, 区别在于这次是kernel赛, 不能使用外部数据和时间限制, 这使得一些技巧变得十分重要. 开始的时候我觉得这次比赛会更加拼单模的表现, 在看了前排的分享之后我发现模型融合还是比较重要的, 一些技巧(比如动态sequences padding)使得模型可以更快速收敛, 得到更多的模型进行融合. 相比之下模型结构的创新在这次比赛中较少, 有之前Toxic的经验, 大家的起点都是比较好的单模.

还有一个值得注意的地方就是public kernel中存在的bug. 人都会犯错, 写kernel的也不一定都是高手, 有时候会存在一些bug, 如果直接fork也不检查, 那后果只能自负. 我个人发现的错误有预处理时步骤颠倒: 一大段拼写替换后再进行特殊字符处理, 这样许多特殊字符在之前已经被清洗掉了. 赛后还有人发现了更加高层次的错误, 可参考此贴: [Common pitfalls of public kernels](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/79911). 所以说fork一时爽, 一直fork不一定一直爽啊.

然后还是老生常谈的trust CV了. public LB的数据仅有5万6, 一点小小的变动就会使得public LB分数产生较大变动, 比如我在预处理的时候仅仅删去了部分特殊字符, local CV只有0.0006的提升, 结果public LB提升了0.005. 不被public LB所迷惑是每个kaggler都需要面对的问题 ;)

最后, Pytorch真香.