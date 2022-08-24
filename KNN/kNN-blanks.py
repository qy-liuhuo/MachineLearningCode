
"""
用途：教学
sqx

"""

"""
函数说明：装载数据集

Parameters:
      fileName - 文件名
    
Returns:
    group - 数据集
    labels - 分类标签

"""


def loadDataSet(fileName):
    # 数据矩阵
    group = []
    # 标签向量
    labels = []
    # 打开文件
    fr = open(fileName)
    # 逐行读取
    for line in fr.readlines():
        # 去掉每一行首尾的空白符，例如'\n','\r','\t',' '
        # 将每一行内容根据'\t'符进行切片
        lineArr = line.strip().split('\t')
        # 添加数据(100个元素排成一行)
        group.append([float(lineArr[0]), float(lineArr[1])])
        # 添加标签(100个元素排成一行)
        labels.append(float(lineArr[2]))
    return group, labels

"""
函数说明：曼哈顿距离求解

Parameters:
	x1 - 向量1
	x2 - 向量2
    
Returns:
    dist - x1与x2间的曼哈顿距离

"""


def distManhattan(x1, x2):
    dist=0;
    for i in range(len(x1)):
        dist+=abs(x1[i]-x2[i])
    return dist



"""

函数说明：kNN算法，分类器

Parameters:
    inX - 用于分类的数据（测试集）
    dataSet - 用于训练的样本特征（训练集）
    labels - 分类标准
    k - kNN算法参数，选择距离最小的k个点
    
Returns:
    predClass - 分类结果
    

"""


def classifyKNN(inX, dataSet, labels, k):
    l=[];
    for i in range(len(dataSet)):
        l.append((distManhattan(inX,dataSet[i]),i))
    l.sort()
    yes=0
    for i in range(k):
        if labels[l[i][1]]==1 :
            yes+=1
    
    return yes>=k/2


def main():
    group,labels=loadDataSet("./TrainingSet.txt")
    # 设置测试数据test
    test_class = classifyKNN([0.45,0.1], group, labels, 3)
    # 打印预测结果
    print(test_class)
if __name__ == '__main__':
    main()
