# 图神经网络基础

节点有节点的属性，边有边的属性

节点可以分为：labeled node和unlabeled node。

图卷积：

1. graph >> spatial-based convolution(空间卷积)
2. Fourier Domain(傅里叶变化) ; Spectral-based convolution

常见字母代表：

I：单位矩阵

A：邻接矩阵

E：边特征矩阵

F：节点特征矩阵

## 路线图

![学习路线图](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/image-20210301160011304.png)

重点：GAT（graph attention network），GCN







![术语](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/image-20210301160546719.png)



![NN4G](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/image-20210301162702318.png)



![NN4G的readout过程](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/image-20210301160956244.png)

![DCNN过程](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/image-20210301162910303.png)

![DCNN输出](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/image-20210301162949574.png)



![MoNET](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/image-20210301163052141.png)

![GAT](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/image-20210301162159253.png)



![建议的update方式](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/image-20210301163212595.png)



拉普拉斯矩阵L = 度矩阵D - 邻接矩阵A

L是半正定，特征值都是大于等于0。Lf中的第i个结果代表了第i点和相邻节点的差。

f<sup>T</sup>Lf代表了两个点的能量差异，可以当作一种“能量”，“频率”来使用。

特征值可以表示频率大小，特征值越大，差异越大。

频率越大，相邻点的信号变化量越大



![谱图理论](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/image-20210301205522839.png)

![ChebNet](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/image-20210301210339879.png)



![转换成切比雪夫多项式的基底](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210301210846467.png)

![GCN公式推导](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/image-20210302193942635.png)



## 各种任务

![任务和数据集](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/image-20210301160240110.png)

![分类](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/image-20210302194429019.png)

![回归](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/image-20210302194449410.png)

![边分类](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/image-20210302194633097.png)



![总结](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/image-20210302195551939.png)



GatedGCN效果不错？

