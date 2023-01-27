# 从零开始搭建神经网络-ANN
[![Open in Colab complete](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CjRovA7WWdQ0mFFtlf2YtjoqladXxvcJ?usp=sharing)

[博客地址](http://lovemefan.top/blog/2020/11/12/%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%80)


## 1.1 神经网络

每层都有若干个节点，每个节点就好比一个神经元（neuron），它与上一层的每个节点都保持着连接，且它的输入是上一层每个节点输出的线性组合。每个节点的输出是其输入的函数，把这个函数叫激活函数（activation function）。人工神经网络通过“学习”不断优化那些线性组合的参数，它就越有能力完成人类希望它完成的目标。

![image-20201112195759523](https://pan-lovemefan.oss-cn-shenzhen.aliyuncs.com/img/image-20201112195759523.png)

除了输入层（第1层）以外，第 l+1 层第i个节点的输入为：

$ z^{(l+1)}_i = W^{(l)}_{i1}a^{(l)}_{1} + W^{(l)}_{i2}a^{(l)}_{2}+ \dots +W^{(l)}_{is_l}a^{(l)}_{s_l} + b^{(l)}_{i}$ 

其中$s_l$表示第$l$层的节点数

第$l+1$层第和i个节点的输出为

$a^{(l+1)}_{i} = f(z^{(l+1)}_{i}) $



## 1.2 ESC-50 语音分类数据集

下载数据集https://github.com/karoldvl/ESC-50/archive/master.zip

其中有四个文件：

数据集中有2000个样本



## 1.3 正向传播

设$X$为样本空间， $Y=\{y_0, y_1 ,...,y_9 \}$ 为标签

样本$(x,y) x\in X ,y \in Y$

其中$x=\{ x_1,x_2,...,x_{784}\}$ , $y=\{y_i \},i \in (0,9]$

模型的参数$\theta = \{ \theta_0,\theta_1,\theta_2 \}$

其中$\theta_0=\{b_1,b_2,...,b_{784}\}, \theta_1=\{ w^{(1)},b^{(1)}\},\theta_2=\{w^{(2)},b^{(2)}\} $



### 1.3.1 模型结构

$$
\begin{gathered}
layer_0 = input[784] + b_0[784] &\\
layer_1 = tanh(layer_0[784]*w^{(1)}[784,1024] + b^{(1)}) &\\
layer_2 = softmax(layer_1[1024]*w^{(2)}[1024,10] + b^{(2)})& \\
\end{gathered}
$$

### 1.3.2 公式推导：

$$
x_{in} = x + b_0 \\
z^{(1)} = w^{(1)}x + b^{(1)}\\
a^{(1)} = tanh(z^{(1)})\\
z^{(2)} = w^{(2)}a^{(2)} + b^{(2)}\\
y_{pred} = a^{(2)} = sofmax(z^{(2)})\\
Loss = \left \| y_{pred} - y  \right \|_2
$$



## 1.4 反向传播

### 1.4.1 公式推导

$$
\frac{\partial Loss}{\partial w^{(2)}} = \frac{\partial Loss}{\partial y_{pred}} \times \frac{\partial y_{pred}}{\partial z^{(2)}}\times\frac{\partial z^{(2)}}{\partial w^{(2)}}=2(y_{pred} - y)\partial softmax(z^{(2)})a^{(1)} \\
\frac{\partial Loss}{\partial b^{(2)}} = \frac{\partial Loss}{\partial y_{pred}} \times \frac{\partial y_{pred}}{\partial z^{(2)}}\times\frac{\partial z^{(2)}}{\partial b^{(2)}}=2(y_{pred} - y)\partial softmax(z^{(2)}) \\
\frac{\partial Loss}{\partial w^{(1)}} = \frac{\partial Loss}{\partial y_{pred}} \times \frac{\partial y_{pred}}{\partial z^{(2)}}\times \frac{\partial z^{(2)}}{\partial a^{(1)}} \times  \frac{\partial a^{(1)}}{\partial z^{(1)}} \times  \frac{\partial z^{(1)}}{\partial w^{(1)}}=2(y_{pred} - y)\partial softmax(z^{(2)})w^{(2)}\partial tanh(z^{(1)})x_{in} \\
\frac{\partial Loss}{\partial b^{(1)}} = \frac{\partial Loss}{\partial y_{pred}} \times \frac{\partial y_{pred}}{\partial z^{(2)}}\times \frac{\partial z^{(2)}}{\partial a^{(1)}} \times  \frac{\partial a^{(1)}}{\partial z^{(1)}} \times  \frac{\partial z^{(1)}}{\partial b^{(1)}}=2(y_{pred} - y)\partial softmax(z^{(2)})w^{(2)}\partial tanh(z^{(1)})\\
\frac{\partial Loss}{\partial b^{(0)}} = \frac{\partial Loss}{\partial y_{pred}} \times \frac{\partial y_{pred}}{\partial z^{(2)}}\times \frac{\partial z^{(2)}}{\partial a^{(1)}} \times  \frac{\partial a^{(1)}}{\partial z^{(1)}} \times  \frac{\partial z^{(1)}}{\partial x_{in}} = 2(y_{pred} - y)\partial softmax(z^{(2)})w^{(2)}\partial tanh(z^{(1)})w^{(1)}
$$

其中

$\partial softmax(x_i)= softmax(x_i)(1-softmax(x_i))$

$softmax(x_i) = \frac{e{x_i}}{\sum_j^{n} e^{x_j}}$

$\partial tanh(x)= 1 - tanh^2(x)$

$tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

## 1.4.2 更新参数

$$
w^{(2)} = w^{(2)} - \alpha \frac{\partial Loss}{\partial w^{(2)}}\\
b^{(2)} = b^{(2)} - \alpha \frac{\partial Loss}{\partial b^{(2)}}\\
w^{(1)} = w^{(1)} - \alpha \frac{\partial Loss}{\partial w^{(1)}}\\
b^{(1)} = b^{(b)} - \alpha \frac{\partial Loss}{\partial b^{(1)}}\\
b^{(0)} = b^{(0)} - \alpha \frac{\partial Loss}{\partial b^{(0)}}
$$