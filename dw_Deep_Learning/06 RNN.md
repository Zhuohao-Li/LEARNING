# notes5 RNN

为什么需要 CNN？ -前面介绍的全连接神经网络权重矩阵的参数非常多，很多是没有必要的。比如处理图像的时候，图像中的物体，都具有局部不变性特征，即尺度缩放、平移、旋转等操作不影响其语义信息，但是全连接前馈网络很难提取这些局部不变特征。CNN 启发于生物学上的感受野，提高了神经网络的效率。

### 计算图

##### 概念

为什么引入计算图？

计算图的引入是为了更方便表示网络，用来描述计算结构。它包含 node 和 vertex， nodes 通常作为变量，而 vertex 通常来表示函数。 这和信号与系统中的系统很像。

nodes 就是系统的输入输出，而 vertex 代表着对于输入输出的一种操作。通常作为函数来表示

## RNN

History： 比较出名的几个： LSTM， BRNN

#### construction

（图见 datawhale 讲义）
其中各个符号的表示：$x_t,s_t,o_t$分别表示的是$t$时刻的输入、记忆和输出，$U,V,W$是 RNN 的连接权重，$b_s,b_o$是 RNN 的偏置，$\sigma,\varphi$是激活函数，$\sigma$通常选 tanh 或 sigmoid，$\varphi$通常选用 softmax。

其中 softmax 函数，用于分类问题的概率计算。本质上是将一个 K 维的任意实数向量压缩 (映射)成另一个 K 维的实数向量，其中向量中的每个元素取值都介于(0，1)之间。

简单来说，t 时刻的记忆=t 时刻的输入与 t-1 时刻的输入的加权和与偏置和的激活函数
t 时刻的输出=t 时刻的记忆与偏置的加权和的选择函数（softmax）

$$
\sigma(\vec{z})_{i}=\frac{e^{z_{i}}}{\sum_{j=1}^{K} e^{z_{j}}}
$$

#### common construction

RNN 本质上就是同一个网络多次反复使用

- Elman Network
  最经典的 RNN 结构

- Jordan Network
  在 Elman 的基础上进行交叉

- 各种 RNN

### training RNN: BPTT

BP 算法，就是定义损失函数 Loss 来表示输出 $\hat{y}$ 和真实标签 y 的误差，通过链式法则自顶向下求得 Loss 对网络权重的偏导。沿梯度的反方向更新权重的值， **直到 Loss 收敛**。而这里的 BPTT 算法就是加上了时序演化，后面的两个字母 TT 就是 Through Time。

输出函数定义为：

$$
\begin{array}{l}s_{t}=\tanh \left(U x_{t}+W s_{t-1}\right) \\ \hat{y}_{t}=\operatorname{softmax}\left(V s_{t}\right)\end{array}
$$

这和我们前面讨论的是类似的

损失函数定义为：

$$
\begin{aligned} E_{t}\left(y_{t}, \hat{y}_{t}\right) =-y_{t} \log \hat{y}_{t} \\ E(y, \hat{y}) =\sum_{t} E_{t}\left(y_{t}, \hat{y}_{t}\right) \\ =-\sum_{t} y_{t} \log \hat{y}_{t}\end{aligned}
$$

E 分别对 V，W，U 求梯度优化

### 长短时记忆网络

RNN 一个不可忽略的问题就是梯度消失问题，梯度消失的原因有两个：BPTT 算法和激活函数 Tanh

有两种解决方案，分别是 ReLU 函数和门控 RNN(LSTM).

##### LSTM

ReLU 自不用多说，LSTM 是一种用于深度学习领域的人工循环神经网络（RNN）结构。一个 LSTM 单元由输入门、输出门和遗忘门组成，三个门控制信息进出单元。

- LSTM 依靠贯穿隐藏层的细胞状态实现隐藏单元之间的信息传递，其中只有少量的线性操作
- LSTM 引入了“门”机制对细胞状态信息进行添加或删除，由此实现长程记忆
- “门”机制由一个 Sigmoid 激活函数层和一个向量点乘操作组成，Sigmoid 层的输出控制了信息传递的比例

### RNN 应用

RNN 在 NLP 领域应用广泛，比如 Word Embedding：自然语言处理(NLP)中的一组语言建模和特征学习技术的统称，其中来自词汇表的单词或短语被映射到实数的向量。

- 图像描述
- 自动写作
- 机器翻译
- 自动作曲
- 语言模型
