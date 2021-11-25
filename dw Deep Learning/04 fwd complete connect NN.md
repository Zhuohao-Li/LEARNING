# notes5 CNN

为什么需要 CNN？ -前面介绍的全连接神经网络权重矩阵的参数非常多，很多是没有必要的。比如处理图像的时候，图像中的物体，都具有局部不变性特征，即尺度缩放、平移、旋转等操作不影响其语义信息，但是全连接前馈网络很难提取这些局部不变特征。CNN 启发于生物学上的感受野，提高了神经网络的效率。

### convolution

##### 概念

在信号与系统，复变函数，数理方法或者 DSP 中，我们已经学过 conv 的数学表达式：

$$
(f * g)(n)=\int_{-\infty}^{\infty} f(\tau) g(n-\tau) d \tau
\\
n = \tau + (n - \tau)
\\
(f * g)(n) = \sum_{\tau = -\infty}^{\infty} f(\tau) g(n-\tau)
$$

在信号与系统中，卷积通常用来处理一个数字信号处理系统的输入，通过与系统函数卷积，得到输出。conv 本质上干的就是信号 **延迟累积** 的效果。

例如下面这个例子：

例如，假设一个信号发生器每个时刻 $t$ 产生一个信号 $x_t$ ，其信息的衰减率为 $w_k$ ，即在 $k−1$ 个时间步长后，信息为原来的 $w_k$ 倍，假设 $w_1 = 1,w_2 = 1/2,w_3 = 1/4$，则时刻 $t$ 收到的信号 $y_t$ 为当前时刻产生的信息和以前时刻延迟信息的叠加，即：

$$
\begin{aligned} y_{t} &=1 \times x_{t}+1 / 2 \times x_{t-1}+1 / 4 \times x_{t-2} \\ &=w_{1} \times x_{t}+w_{2} \times x_{t-1}+w_{3} \times x_{t-2} \\ &=\sum_{k=1}^{3} w_{k} \cdot x_{t-k+1} \end{aligned}
$$

其中 $w_k$ 就是滤波器，也就是常说的卷积核 convolution kernel。

因此，一般的表述为：给定一个输入信号序列 $x$ 和滤波器 $w$，卷积的输出为：

$$
y_t = \sum_{k = 1}^{K} w_k x_{t-k+1}
$$

filter（滤波器）的不同决定了提取的输入信号的不同特征：比如 低通滤波器，高通滤波器等

filter 的滑动步长 S (stride) 定义为 filter 在输入上遍历的最小单位，离散时间系统上 S 是一个大于等于 1 的正整数。零填充 P (padding) 通常在要求输入数据和输出数据长度相等时，会给输入数据额外加长添 0

卷积的结果按输出长度不同可以分为三类：

1. 窄卷积：步长 𝑇 = 1 ，两端不补零 𝑃 = 0 ，卷积后输出长度为 𝑀 − 𝐾 + 1
2. 宽卷积：步长 𝑇 = 1 ，两端补零 𝑃 = 𝐾 − 1 ，卷积后输出长度 𝑀 + 𝐾 − 1
3. 等宽卷积：步长 𝑇 = 1 ，两端补零 𝑃 =(𝐾 − 1)/2 ，卷积后输出长度 𝑀

$M$是输入序列的长度，$K$是卷积序列的长度。

早期 CNN 论文，卷积默认为窄卷积，目前，一般认为是等宽卷积

图像处理中，需要解释一下二维卷积 [二维卷积](https://blog.csdn.net/appleyuchi/article/details/78597516?ops_request_misc=&request_id=&biz_id=102&utm_term=%E4%BA%8C%E7%BB%B4%E5%8D%B7%E7%A7%AF&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-78597516.first_rank_v2_pc_rank_v29&spm=1018.2226.3001.4187)

In Pytorch:

- in_channels：输入的通道数
- out_channels：输出的通道数
- kernel_size：卷积核大小
- stride：步长
- padding：对输入进行填充
- dilation：卷积核的空洞
- groups：卷积的分组
- bias：卷积结果后添加的偏置
- padding_mode：填充的类型

**转置卷积** **空洞卷积**

## CNN

#### convolution layer

就像之前讲的，conv 的主要步骤为：

- input layer 根据 padding 补 0
- kernel transparent
- 根据 stride （=1， =2） 扫过 input layer matrix

根据 sweep 的起始位置不同，分成以下模式：

- full
- same
- valid

我们再回顾一下，conv 的维度计算，这样的维度计算在今后的 research 当中还是非常关键的！一定要迅速和准确！
input layer： n
kernel：f
output layer： n-f+1
如果在 input 周围以 padding=p 补 0，output： n+**2p**-f+1

#### 感受野 receptive field

卷积神经网络每一层输出的特征图(featuremap)上的像素点在输入图片上映射的区域大小，即特征图上的一个点对应输入图上的区域。

rf 的大小和前面一层的 kernel 大小和 stride 有关，也和前面一层的 rf 有关。

$$
 rf_{i} = (rf_{i+1} - 1) \tims s_i + K_i
$$

$s_i$是第 i 层的步长 $K_i$是第 i 曾 kernel 的大小

###### convolution layer depth = conv 的个数

一个卷积层通常包含多个尺寸一致的卷积核

## activation

激活函数是用来加入非线性因素，提高网络表达能力，卷积神经网络中最常用的是**ReLU**，Sigmoid 使用较少。

ReLU 函数的优点：

- 计算速度快，ReLU 函数只有线性关系，比 Sigmoid 和 Tanh 要快很多
- 输入为正数的时候，不存在梯度消失问题

ReLU 函数的缺点：

- 强制性把负值置为 0，可能丢掉一些特征
- 当输入为负数时，权重无法更新，导致“神经元死亡”(将学习率设置为一个较小的值)

PReLU (Parametric ReLU)

**2. Parametric ReLU**

$$
f(x)=\left\{\begin{array}{l}\alpha x, x<0 \\ x, x \geq 0\end{array}\right.
$$

- 当 𝛼=0.01 时，称作 Leaky ReLU
- 当 𝛼 从高斯分布中随机产生时，称为 Randomized ReLU(RReLU)

**PReLU**函数的优点：

- 比 sigmoid/tanh 收敛快
- 解决了 ReLU 的“神经元死亡”问题

**PReLU**函数的缺点：需要再学习一个参数，工作量变大

**3. ELU 函数**

$$
f(x)=\left\{\begin{array}{l}\alpha (e^x-1), x<0 \\ x, x \geq 0\end{array}\right.
$$

**ELU**函数的优点：

- 处理含有噪声的数据有优势
- 更容易收敛

**ELU**函数的缺点：计算量较大，收敛速度较慢

- CNN 在卷积层尽量不要使用 Sigmoid 和 Tanh，将导致梯度消失。
- 首先选用 ReLU，使用较小的学习率，以免造成神经元死亡的情况。
- 如果 ReLU 失效，考虑使用 Leaky ReLU、PReLU、ELU 或者 Maxout，此时一般情况都可以解决

##### 特征图

- 浅层卷积层：提取图像的基本特征，比如边缘，方向，纹理等
- 深层卷积图，提取图像的高阶特征，进行识别

#### 池化层 Pooling

池化过程在一般卷积过程后。池化（pooling） 的本质，其实就是采样。Pooling 对于输入的 Feature Map，选择某种方式对其进行降维压缩，以加快运算速度。

采用较多的一种池化过程叫最大池化（Max Pooling），池化过程类似于卷积过程，表示的就是对一个 feature map 邻域内的值，用一个的 filter，步长为 2 进行‘扫描’，选择最大值输出到下一层，这叫做 Max Pooling。

max pooling 常用的 $stride=2, filter=2$ 的效果：特征图高度、宽度减半，通道数不变。

还有一种叫平均池化（Average Pooling）,就是从以上取某个区域的最大值改为求这个区域的平均值，表示的就是对一个 feature map 邻域内的值，用一个 filter，步长为 2 进行‘扫描’，计算平均值输出到下一层，这叫做 Mean Pooling。

【池化层没有参数、池化层没有参数、池化层没有参数】 （重要的事情说三遍）

池化的作用：

（1）保留主要特征的同时减少参数和计算量，防止过拟合。

（2）invariance(不变性)，这种不变性包括 translation(平移)，rotation(旋转)，scale(尺度)。增强网络对输入图像中的小变形、扭曲、平移的鲁棒性(输入里的微 小扭曲不会改变池化输出——因为我们在局部邻域已经取了最大值/ 平均值)

(3) 帮助我们获得不因尺寸而改变的等效图片表征。这非常有用，因为 这样我们就可以探测到图片里的物体，不管它在哪个位置

Pooling 层说到底还是一个特征选择，信息过滤的过程。也就是说我们损失了一部分信息，这是一个和计算性能的一个妥协，随着运算速度的不断提高，我认为这个妥协会越来越小。

现在有些网络都开始少用或者不用 pooling 层了。

#### output layer

对于分类问题：使用**Softmax**函数

$$
y_i = \frac{e^{z_i}}{\sum_{i = 1}^{n}e^{z_i}}
$$

对于回归问题：使用线性函数

$$
y_i = \sum_{m = 1}^{M}w_{im}x_m
$$

#### 全连接层

- 对二维的特征图进行降维
- 将学到的特征表示映射到样本标记空间的空间

#### conv training

训练的步骤如下：

1. 用随机数初始化 Kernel 的 weight
2. **Fwd** 输入是训练的图片，依次通过各层 hidden layer。 包括 **卷积**， **ReLU activation**， **pooling**， **全连接层** ，计算每个类别的对应输出效率
3. 计算输出层的总误差
4. **bwd** 计算 **梯度** ，利用梯度下降法更新所有 weight 的 值

   attention： Kernel 的个数，Kernel 的 size，网络架构（NN 的连接方式）是不会在训练过程中改变的，只有 Kernel matrix 和 weight 进行更新

what is subsampling:
Subsampling is a method that reduces data size by selecting a subset of the original data. The subset is specified by choosing a parameter n, specifying that every nth data point is to be extracted. For example, in structured datasets such as image data and structured grids, selecting every nth point produces the results shown in Fig. 1.36. Subsampling modifies the topology of a dataset. When points or cells are not selected, this leaves a topological “hole.” Dataset topology must be modified to fill the hole. In structured data, this is simply a uniform selection across the structured i-j-k coordinates. In structured data, the hole must be filled in by using triangulation or other complex tessellation schemes. Subsampling is not typically performed on unstructured data because of its inherent complexity.

## classic CNN

#### LeNet-5

- input: 32\*32
- c1: 6 5\*5 kernels
- s2: average pooling 2\*2
- c3: 1 5\*5 kernel
- s4: average pooling 2\*2
- c5: 1 5\*5 kernel, 从此特征图的大小就为 1
- F6: 84 nodes
- output：10 nodes (0:9) 如果节点 i 的输出为 0，那么意味着识别的结果就是 i，采用径向基函数(RBF)的网络连接方式

总结：卷积核大小、卷积核个数(特征图需要多少个)、池化核大小(采样率多少)这些参数都是变化的，这就是所谓的 CNN 调参，需要学会根据需要进行不同的选择。

#### ALexNet

特点：

- 首次成功应用 ReLU 作为 CNN 的激活函数
- 使用 Dropout 丢弃部分神元，避免了过拟合
- 使用重叠 MaxPooling(让池化层的步长小于池化核的大小)， 一定程度上提升了特征的丰富
- 使用 CUDA 加速训练过程
- 进行数据增强，原始图像大小为 256×256 的原始图像中重 复截取 224×224 大小的区域，大幅增加了数据量，大大减 轻了过拟合，提升了模型的泛化能力

#### ResNet [李沐](https://www.bilibili.com/video/BV1P3411y7nn?from=search&seid=14656720868796506038&spm_id_from=333.337.0.0)

随着卷积网络层数的增加，误差的逆传播过程中存在的梯 度消失和梯度爆炸问题同样也会导致模型的训练难以进行，甚至会出现随着网络深度的加深，模型在训练集上的训练误差会出现先降低再升高的现象。残差网络的引入则有助于解决梯度消失和梯度爆炸问题。

残差块

ResNet 的核心是叫做残差块(Residual block)的小单元， 残差块可以视作在标准神经网络基础上加入了跳跃连接(Skip connection)
