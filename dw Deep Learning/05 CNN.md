# notes4 前馈神经网络

# 前馈神经网络

### 神经元模型

##### 对神经元的建模

###### ·MP model

一个神经元包含了一个核心，用来处理和计算（这个核心通常存放了一个 activation function 和 loss function),神经元包括若干个输入和输出，输入给一个神经元的数据有一个 weight, 神经元本身可以有偏置.
"在 M-P 模型中，神经元接受其他 n 个神经元的输入信号(0 或 1)，这些输入信号经过权重加权并求和，将求和结果与阈值(threshold) θ 比较，然后经过激活函数处理，得到神经元的输出。" 这是 paper 的规范表达。因此在后续深度学习的时候，会有一个面向 client 的 parameter 阈值，通过这个 para 可以选择神经网络的 pruning ratio

###### 网络结构

前馈网络，记忆网络，图网络

##### 感知器

###### 单层感知器

能够通过网络自主训练自动确定一些 para。有监督学习。数学上的思想也很简单，就是给定一组输入的期望输出，然后每次得到一组实际输出后，计算实际输出和期望输出的差值，将这个差值乘学习率得到需要修正的百分比，用这个百分比乘输入，然后就能得到一个修正过的值。

基本思路：实际输出和期望输出不相等时，自动调整 weight 和 threshold 的值

具体一点：

- 实际输出比期望输出小，说明训练的时候不够带感吗，没能达到预期效果，训练不足，这时候需要减小 threshold，并且增大=1 的 weight
- 相反的，实际输出比预期输出更小，说明训练的时候过度了，应该提高 threshold，并且降低=1 的 weight

###### 多层感知器

单层感知器只能解决线性可分问题，而不能解决线性不可分问题；为了解决线性不可分问题，我们需要使用多层感知器。

线性可分，直观意义上来讲就是用一个线性函数分割数据

多层感知器指的是由多层结构的感知器递阶组成的输入值向前传播的网络，也被称为前馈网络或正向传播网络。 这在 ResNet 简单的模型训练当中是很常见的，需要 fwd traing 一遍，然后 bwd traing 一遍

包括：

- **Input layer**
- **Hidden layer**
- **output layer**

##### BP algorithm

大名鼎鼎的 Error Back Propagation

这是干什么的呢，其实很简单，就是 fwd training 后反向进行梯度下降，减小误差。误差反响逐层传递，网络的期望输出与实际输出之差的误差信号由输出层经过隐含层逐层向输入层传递。整个神经网络模型就是 fwd + bwd 反复进行的过程，前者是用来进行计算得到 实际输出， 后者是进行误差的反向梯度下降

BP 算法就是通过比较实际输出和期望输出得到误差信号，把误差信 号从输出层逐层向前传播得到各层的误差信号，再通过调整各层的连接权重以减小误差。权重的调整主要使用梯度下降法：

大名鼎鼎的 **梯度下降法** **gradient decreasing**

其实思想很朴素，就是中学里的极值问题！每一层得到的 error 对每一个 weight 求偏导，这样的一次一阶计算过后得到的误差将会更小。

**activation function**

classic：

- Sigmoid: $f(u)=\cfrac{1}{1+e^{-u}}$
- **ReLU**

## an example for BP algorithm

pre-request:

- activation: Sigmoid
- Loss : quadratic loss, loss= $\cfrac{1}{2} (\hat{y} -y)^2$
- 2-layer (1 hidden layer)
- using matrix algebra

output: Y, Y=f(U), f is activation, U is the hidden layer variation.
$\sum_{j=1}^{m} w_{2j1}z_j$, 这可以写成矩阵代数

$\cfrac{\partial E}{\partial w_{2j1}} =  \cfrac{\partial E}{\partial y} \cfrac{\partial y}{\partial u_{21}} \cfrac{\partial u_{21}}{\partial w_{2j1}}$

$= -(\hat{y} - y) (y)(1-y) z_j$

然后计算 gradient:

$\Delta w_{2j1} = \alpha (\hat{y} - y)y(1-y) z_j$

adjust 输入层与中间层的 weight:

$\cfrac{\partial E}{\partial w_{1j1}} =  \cfrac{\partial E}{\partial y} \cfrac{\partial y}{\partial u_{21}} \cfrac{\partial u_{21}}{\partial w_{1j1}}$

而

$cfrac{\partial u_{21}}{\partial w_{1j1}} = \cfrac{\partial{u_{21}}}{\partial z_j} \cfrac{\partial z_j}{\partial u_{1j}} \cfrac{\partial u_{1j}}{\partial w_{1j1}} = -(\hat{y} - y) y(1-y) w_{2jq1} z_j(1-z_j) x_i$
