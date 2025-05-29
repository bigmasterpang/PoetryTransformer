![image](https://github.com/user-attachments/assets/2d6a306d-4b8b-49a4-a683-8f26fd442abf)# PoetryTransformer
基于Transformer架构的智能古诗生成系统，能够根据用户输入的上联自动生成意境相符的下联诗句。本项目融合了深度学习与传统文化，实现了高质量的古诗创作能力。

# 数据集生成
git clone https://github.com/chinese-poetry/chinese-poetry
获取诗词，运行process_dataset.py进行处理

![image](https://github.com/user-attachments/assets/f57c6ba8-d75f-4836-a9c8-a641c763ffc8)


# 模型训练
运行transformer_train.py进行训练
模型很简单，就是一个多层transformer，尝试过增加残差连接，效果很差，还不如现在这个模型。
这个模型AI告诉我加入一个语义感知层，效果比只有transformer要好一些，至少看上去更像古诗。

![image](https://github.com/user-attachments/assets/c74dba84-293a-49cb-a62f-ee17add6c42f)

# 推理
调用generate_poetry.py进行推理，只要配置好模型文件位置

![image](https://github.com/user-attachments/assets/9a797e55-8131-4bb3-a314-b877c7e36793)

# 感悟
这个是一个基础对transformer理解的小demo，可以加深你对transformer的理解，包括文字编码、位置编码，网络的连接设计。
本项目大部分都是基于ai编写的代码。
设计之初，为网络设计了很多规则来优化Loss，包括长句子的限制、重复的惩罚等，效果不是很好。
后来参照其他人的相关代码，深入理解了一下，发现他们使用的网络都是最普通的网络。
所以，回到对transformer的最初理解，只需要设计好输入和输出的格式，剩下的交给网络吧，听天由命。

训练之后的效果，还凑合，至少看着像一首狗屁不通的诗。
