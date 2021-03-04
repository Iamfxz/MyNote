

# Contrastive Multi-View Representation Learning on Graphs

对比图的结构视图来学习节点和图级表示的自监督方法





## 一、相关工作

### Random walks

### Graph kernels

heat kernel

Personalized PageRank

### Graph auto encoders (GAE)

![img](https://pic1.zhimg.com/80/v2-d9c5e951f11f291f5ccb133a2891b4d0_1440w.jpg)

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BZ%7D+%3D+%5Cmathrm%7BGCN%7D%28%5Cmathbf%7BX%7D%2C+%5Cmathbf%7BA%7D%29+%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathrm%7BGCN%7D%28%5Cmathbf%7BX%7D%2C+%5Cmathbf%7BA%7D%29+%3D+%5Ctilde%7B%5Cmathbf%7BA%7D%7D+%5Cmathrm%7BReLU%7D+%28%5Ctilde%7B%5Cmathbf%7BA%7D%7D%5Cmathbf%7BXW_0%7D%29%5Cmathbf%7BW_1%7D+%5C%5C)

### Deep graph Infomax(DGI)

#### 自编码器

要求保留原始数据尽可能多的**重要信息**。

- 第一想法：传统自编码器，用隐藏向量还原原始数据，即训练目标为output拟合原始数据

![img](https://bkimg.cdn.bcebos.com/pic/4d086e061d950a7b988f021904d162d9f3d3c9b1?x-bce-process=image/watermark,image_d2F0ZXIvYmFpa2UxODA=,g_7,xp_5,yp_5/format,f_auto)

- 进一步想法：变分自编码器，为每个样本构造专属的正态分布，然后采样获得隐藏向量来重构。隐藏向量的分布尽量能接近高斯分布，能够随机生成隐含变量喂给解码器，也提高了泛化能力。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181128111509647.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMxODk1OTQz,size_16,color_FFFFFF,t_70)

- 但是，对于数据集和任务来说，完成任务所需要的特征并不一定要能完成图像重构。例如，辨别百元假钞不一定要能完整复刻出百元假钞。



#### 互信息（MI, mutual information）

好特征的基本原则应当是**“能够从整个数据集中辨别出该样本出来”**，也就是说，提取出该样本（最）**独特**的信息。

熵H(Y)与条件熵H(Y|X)之差称为互信息，决策树学习中的信息增益等价于训练数据集中类与特征的互信息。

互信息：变量间**相互依赖性**的量度。不同于相关系数，互信息并不局限于实值随机变量。它能度量两个事件集合之间的相关性。

- 用 X 表示原始图像的集合，用 x∈X 表示某一原始图像。

- Z 表示编码向量的集合，z∈Z 表示某个编码向量。

- p(z|x) 表示 x 所产生的编码向量的分布，我们设它为高斯分布。这是我们要找的**编码器**。

![img](https://pic3.zhimg.com/80/v2-8a4a3dd7b5b75e2160ec0b8130ca1502_1440w.jpg)

- p̃(x) 原始数据的分布，p(z) 是在 p(z|x) 给定之后整个 Z 的分布

![img](https://pic1.zhimg.com/80/v2-6c2b5c4769d44a82439dbe2c934a5fc4_1440w.jpg)

- 好的特征编码器，应该要使得互信息尽量地大:

![img](https://pic4.zhimg.com/80/v2-50902c9abfcf086cb253199e31b2322f_1440w.jpg)

- H是信息熵，I是互信息。  
- **I(X, Z) = H(Z) - H(Z|X)**  ：熵 H(Z) 看作一个随机变量不确定度的量度，那么 H(Z|X) 就是 X 没有涉及到的 Z 的部分的不确定度的量度。总的Z的不确定度，减去知道X而剩下的Y的不确定度，所以可以直观地理解互信息是**Z变量提供给Y的信息量**

![img](https://www.omegaxyz.com/wp-content/uploads/2018/08/MI5.png)

- ![[公式]](https://www.zhihu.com/equation?tex=I%28X%3BY%29%3DKL%28p%28x%2Cy%29+%7C%7C+p%28x%29p%28y%29%29)

## 二、模型组件

![image-20210303232919140](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/image-20210303232919140.png)

<img src="https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/image-20210304013905369.png" alt="算法过程" style="zoom:50%;" />

- 增强机制：对图的结构进行增广，然后对相同的节点进行子采样。类似于CV中的裁剪。
- 两个专用的GNN：即图编码器。对应原数据和增强后的数据。
  
- 使用GCN。σ(AXΘ)  and  σ(SXΘ), X为初始节点的特征，Θ为学习参数。
  
- 一个共享的MLP（靠左）：用于学习图的**节点表示**。具有两个隐藏层和PReLU激活函数。

  ![image-20210303201008592](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/image-20210303201008592.png)

- 一个图池化层P：即readout函数，而后传入共享MLP（结构同上）中，得出**图表示**。

  - 每个GCN层中的节点表示的总和，连接起来，然后将它们馈送到一个单层前馈网络。
  - ![image-20210303211111157](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210303211111157.png)
  - 分别求和作为下游任务的  **图表示**  和  **节点表示**
  - ![image-20210303234236283](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/image-20210303234236283.png)

- 鉴别器D：图的**节点表示**和另一个图的**图形表示**进行对比。并对它们的一致性进行评分。

  - 损失函数：最大化互信息
  - ![image-20210303235501903](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/image-20210303235501903.png)
  - ![image-20210304013125246](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/image-20210304013125246.png)




## 三、解决方法

### 广义图传播（diffusion）

![广义图扩散](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/image-20210303185525804.png)

T是广义转移矩阵。从一个状态转移到下一个。

Θ是权重系数，表示全局和局部信息的比例。所有θ之和为1。

### 两种图传播算法的实例（卷积？）

 heat kernel : 带入T = AD<sup>-1</sup>, θ<sub>k</sub> = α（1-α）<sup>k</sup>。α是随机游走的传送概率，t是

Personalized PageRank : 带入T = D<sup>-1/2</sup>AD<sup>-1/2</sup>, θ<sub>k</sub> = e<sup>-t</sup>t<sup>k</sup>/k!

![image-20210303185803768](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/image-20210303185803768.png)

### 子采样

从一个图中随机采样节点及边，然后再从另一个图（扩散后）中确定对应的节点和边。



## 四、总结

图的对比学习与视觉对比学习不同：

1. 扩充view的数量超过两个不会改善性能，最好的效果是**邻接矩阵和传播矩阵**进行对比学习。
2. **对比节点和视图的表示**达到更好的效果，优于，图-图表示对比学习，不同长度的编码对比学习。
3. **简单的图读出层(求和)**比differentiable pooling（DiffPool）效果更好。
4. 预训练时候，应用正则化（提前停止除外）或规范化层会对性能产生负面影响。

