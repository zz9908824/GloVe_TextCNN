# GloVe+TextCNN 文本分类

首先用GloVe训练语料词向量，然后TextCNN基于GloVe学习的词向量实现文本分类。

词向量训练模型地址github：https://github.com/stanfordnlp/GloVe.git  

数据集：cnews 链接: https://pan.baidu.com/s/1Nm4AbkL9M-KkplCFVO6HUw?pwd=pb2b 提取码: pb2b  

glove预训练词向量：链接: https://pan.baidu.com/s/1Wa6GVXVwNTFLsUBXZYAgEg?pwd=3wgi 提取码: 3wgi  

Data文件：TextCNN+GloVe链接: https://pan.baidu.com/s/1QalxPYgASpMwiuqsNzNMgw?pwd=1gwk 提取码: 1gwk **从网盘下载Data文件只需要和仓库中code文件夹放在同级目录下然后修改代码中相应路径即可**  


## 一、实现

### 1.GloVe词向量学习

数据集：cnews 中文新闻分类数据集 是THUCNews的子集，包含10个类别。

（1）数据准备与预处理

拆分cnews为训练集、验证集、测试集

文本清洗并用jieba实现分词，清洗和分词方法将在TextCNN文本分类阶段复用，即文本分类TextCNN和词向量学习GloVe使用相同的清洗和分析方法。

（2）词向量学习

vocab_count.c文件统计经过分词处理后的语料数据中每个单词的词频，并按词频降序对单词排序构建词汇表vocab.txt；

cooccur.c文件统计单词共现，构建单词的共现矩阵。首先根据vocab.txt中的单词构建实现单词和频秩映射的哈希表，定义稠密区准入边界max，当两个单词的频秩积小于max时单词共现权重保存在bigram_table中，大于max时保存在缓冲区cr中。定义窗口大小，统计单词对<word2,word1>的共现频率，并结合两个单词之间的距离计算权重。最后cooccur.c会将所有临时文件经过多路归并合并为一个三元组列表{<word1_id,word2_id>,val},`cooccur.c` 输出有序共现二进制文件（严格按 `word1` 升序、`word2` 升序排列）。**优点：词频分流存储实现分态存储共现数据，内存满足大语料训练；距离权重作为共现权重符合“邻近词语义关联更强”的语言学规律；缓冲区满则做一次内部处理，减少写盘次数。**

shuffle.c文件会打乱单词对顺序。cooccur.c输出的文件是经过排序的文件，如果模型直接使用cooccur.c输出的文件，模型的学习结果会受到词频排序的影响产生偏见，因此需要shuffle.c文件对cooccur.c文件的输出文件做打乱乱序处理。由于cooccur.c文件输出的单词对顺序文件的大小过大，无法一次将完整的文件加载到内存，shuffle.c文件采用了两级“分块打乱+块内打乱”的处理方法。**优点：两级打乱，解决文件大无法一次加载到内存的问题；Fisher-Yates原地洗牌算法（必须保证随机下标j均匀分布），公平无偏的洗牌算法。**    

`glove.c` 文件实现词向量学习。损失函数：

$$
J=\sum_{i,j=1}^{V}f(x_{ij})\big(\boldsymbol{w}_i^\mathrm{T}\tilde{\boldsymbol{w}}_j+b_i+\tilde{b}_j-\log X_{ij}\big)^2
$$

训练完成后会输出词向量文件：`vectors.bin`（二进制文件）与 `vectors.txt`。

### 2.TextCNN文本分类

（1）数据清洗与分词

preprocess_cnews.py文件将cnews数据集按照和GloVe一样的处理方法清洗、分词。

vocab_count.py文件和GloVe统计单词词频并按词频降序排序的思想一致，vocab_count.py文件对分好词的数据集统计单词词频并按词频降序排序得到数据集的单词表文件vocab.txt。

（2）单词和标签映射

transword_id.py文件中`build_vocab(vocab_path, min_count=1)方法`将vocab.txt文件转换为word2id和id2word并打包为pkl。word2id和id2word文件帮助把文本经分词后的单词转换成单词在嵌入层矩阵中的下标位置，从而方便的从模型嵌入层词向量矩阵中取出单词的词向量。

注意：在word2id和id2word前插入<pad>和<unk>，其余单词按照vocab.txt中的顺序转换为word2id和id2word。

word2id和id2word文件的单词顺序和模型嵌入层词向量矩阵单词顺序一致。

label_map.py文件中`load_label_mapping(file_path)方法`将自定义的label_mapping.txt文件转换成字典的格式label2id，帮助把经过清洗和分词的数据文件中的标签转换成对应的数字。

（3）TextCNN文本分类

对比了`预训练词向量加载逻辑（三种模式：基线random / glove_static / glove_finetune）`

random:嵌入层词向量矩阵随机初始化；

glove_static：模型学习过程中，预训练的词向量不改变；

glove_finetune：模型学习过程中，预训练词向量随模型梯度更新变化。

数据加载：段落单词数目定义为500

模型结构：嵌入层(三种模式) -> 卷积层（2 3 4三种卷积核并行，提取多元语义信息）-> 池化和drop -> 分类

损失函数：交叉熵损失

```
1. 数据准备与预处理
   1.1 拆分cnews为训练集、验证集、测试集
   1.2 编写统一的文本清洗+jieba分词函数，全程复用

2. 自训GloVe领域词向量
   2.1 仅用训练集文本，执行统一的分词清洗
   2.2 输入GloVe训练词向量，输出文本格式的词向量文件（词+向量）
   2.3 加载词向量为 {词: numpy向量} 的字典备用

3. 构建词表并保存
   3.1 仅用训练集分词结果，统计每个词的出现频次
   3.2 设置min_count过滤低频词，加入<PAD>、<UNK>特殊token
   3.3 生成word2id、id2word映射字典
   3.4 序列化为vocab.pkl保存

4. 数据标准化与加载
   4.1 对训练/验证/测试集文本：分词 → 转id序列 → 短填长切到固定pad_size
   4.2 封装为Dataset和DataLoader，按批次batch_size加载

5. TextCNN模型构建与初始化
   5.1 实例化Config配置类，设置所有超参数
   5.2 构建嵌入权重矩阵：随机初始化 → <PAD>设为全0 → 按word2id匹配填充GloVe向量
   5.3 初始化Embedding层，加载填充好的权重，设置可微调
   5.4 搭建多尺度卷积层、池化、Dropout、全连接分类层

6. 模型训练与验证
   6.1 定义交叉熵损失、Adam优化器
   6.2 循环训练epoch：
       - 训练集批次前向传播 → 计算损失 → 反向传播更新参数
       - 每个epoch结束后，用验证集评估准确率和损失
       - 保存验证集准确率最高的模型权重(.ckpt)，触发早停则终止训练

7. 最终测试评估
   7.1 加载保存的最佳模型权重
   7.2 在测试集上跑一次前向传播，计算最终的分类准确率、精确率、召回率、F1等指标
   7.3 输出最终结果
```

## 二、结果

random模式模型的学习结果：
<img width="727" height="823" alt="image" src="https://github.com/user-attachments/assets/5fb40054-4b13-4b31-b3d0-edcfb366e86e" />
<img width="735" height="907" alt="image" src="https://github.com/user-attachments/assets/a6381794-8394-4989-a94e-535ede7579fb" />

glove_static模式模型学习结果（连续五轮没有最优的验证集准确率，早停）：
<img width="721" height="1099" alt="image" src="https://github.com/user-attachments/assets/5b0d270c-f126-4d40-82eb-c10f4cb140e2" />
<img width="1030" height="814" alt="image" src="https://github.com/user-attachments/assets/a8408504-a700-4519-9774-32faa432ecdb" />

glove_finetune模式模型学习结果（连续五轮没有最优的验证集准确率，早停）：
<img width="1317" height="1363" alt="image" src="https://github.com/user-attachments/assets/da943677-6769-42cd-b027-8971bd08af07" />
<img width="763" height="679" alt="image" src="https://github.com/user-attachments/assets/3663baf3-c837-4d86-9ac1-fe5800c04573" />

