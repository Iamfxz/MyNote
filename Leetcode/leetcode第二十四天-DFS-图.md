**今天完成题目**:690,1688,733  
690:员工的重要性
- 将员工数据结构转换成字典,从而可以快速找到对应id的员工数据
- 利用dfs的思想,遍历员工及其子员工

1688,733:颜色填充,渲染
- 记得当判断当新旧颜色一致时候直接返回原来的image
- 只需要dfs每个元素的上下左右即可,记得判断边界条件

997:找到小镇的法官
- 依次判断三个条件的成立
- 转化为图的理解是:在有向图中找到一个顶点,其他N-1个联通这个顶点,且指向它(条件2).这个顶点有且仅有一个(条件3).而且这个顶点不指向其他顶点.(条件1)
- 特殊情况下:无信任关系时候,当总人数为1时,他为法官,总人数大于1时,无结果,返回-1
- 通过data.copy()可以拷贝数据,默认为引用.