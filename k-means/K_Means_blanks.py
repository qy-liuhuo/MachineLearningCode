# -*- coding: utf-8 -*-
"""
用途：教学
sqx

"""
import matplotlib.pyplot as plt
import numpy as np

"""
函数说明：将文本文档中的数据读入到python中

Parameters:
    fileName - 文件名
    
Returns:
    dataMat - 数据矩阵

"""


def loadDataSet(fileName):

    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)

    return dataMat


"""
函数说明：数据向量计算距离--欧几里德距离的平方

Parameters:
    vecA - 数据向量A
    vecB - 数据向量B
    
Returns:
    两个向量之间的距离


"""


def distEclud(vecA, vecB):
    return np.sum(np.power(vecA - vecB, 2))


"""
函数说明：随机初始化k个质心（质心满足数据边界之内）

Parameters:
    dataSet - 输入的数据集
    k - 选取k个质心
    
Returns:
    centroids - 返回初始化得到的k个质心向量

"""


def randCent(dataSet, k):
    # 得到数据样本的维度
    n = np.shape(dataSet)[1]
    # 初始化为一个(k,n)的全零矩阵
    centroids = np.mat(np.zeros((k, n)))
    # 遍历数据集的每一个维度
    for j in range(n):
        # 得到该列数据的最小值,最大值
        minJ = np.min(dataSet[:, j])
        maxJ = np.max(dataSet[:, j])
        # 得到该列数据的范围(最大值-最小值)
        rangeJ = float(maxJ - minJ)
        # k个质心向量的第j维数据值随机为位于(最小值，最大值)内的某一值
        # Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    # 返回初始化得到的k个质心向量
    return centroids


"""
函数说明：k-means聚类算法

Parameters:
    dataSet - 用于聚类的数据集
    k - 选取k个质心
    distMeas - 距离计算方法,默认欧氏距离distEclud()
    createCent - 获取k个质心的方法,默认随机获取randCent()
    
Returns:
    centroids - k个聚类的聚类结果
    clusterAssment - 聚类误差


"""


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    # 获取数据集样本数
    m = np.shape(dataSet)[0]
    # 初始化一个（m,2）全零矩阵
    clusterAssment = np.mat(np.zeros((m, 2)))
    # 创建初始的k个质心向量
    centroids = createCent(dataSet, k)
    # 聚类结果是否发生变化的布尔类型
    clusterChanged = True
    # 只要聚类结果一直发生变化，就一直执行聚类算法，直至所有数据点聚类结果不发生变化
    while clusterChanged:
        # 聚类结果变化布尔类型置为False
        clusterChanged = False
        # 遍历数据集每一个样本向量
        for i in range(m):
            # 初始化最小距离为正无穷，最小距离对应的索引为-1
            minDist = float('inf')
            minIndex = -1
            # 循环k个类的质心
            for j in range(k):
                # 计算数据点到质心的欧氏距离
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                # 如果距离小于当前最小距离
                if distJI < minDist:
                    # 当前距离为最小距离，最小距离对应索引应为j(第j个类)
                    minDist = distJI
                    minIndex = j
            # 当前聚类结果中第i个样本的聚类结果发生变化：布尔值置为True，继续聚类算法
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            #######################  填空-sqx  1 #############################
            # 填空-sqx
            # 更新当前变化样本的聚类结果和平方误差
                clusterAssment[i, :] = minIndex, distJI

        # 遍历每一个质心
        for cent in range(k):
            # 将数据集中所有属于当前质心类的样本通过条件过滤筛选出来
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            #######################  填空-sqx  2 #############################
            # 填空-sqx
            # 计算这些数据的均值向量
            # centroids[cent, :] =
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    # 返回k个聚类，聚类结果及误差
    return centroids, clusterAssment


"""
函数说明：绘制数据集

Parameters:
    fileName - 文件名
    
Returns:
    None


"""


def plotDataSet(filename):
    # 导入数据
    datMat = np.mat(loadDataSet(filename))
    # 进行k-means算法其中k为4
    myCentroids, clustAssing = kMeans(datMat, 5)
    clustAssing = clustAssing.tolist()
    myCentroids = myCentroids.tolist()
    xcord = [[], [], [], [],[]]
    ycord = [[], [], [], [],[]]
    datMat = datMat.tolist()
    m = len(clustAssing)
    for i in range(m):
        if int(clustAssing[i][0]) == 0:
            xcord[0].append(datMat[i][0])
            ycord[0].append(datMat[i][1])
        elif int(clustAssing[i][0]) == 1:
            xcord[1].append(datMat[i][0])
            ycord[1].append(datMat[i][1])
        elif int(clustAssing[i][0]) == 2:
            xcord[2].append(datMat[i][0])
            ycord[2].append(datMat[i][1])
        elif int(clustAssing[i][0]) == 3:
            xcord[3].append(datMat[i][0])
            ycord[3].append(datMat[i][1])
        elif int(clustAssing[i][0]) == 4:
            xcord[4].append(datMat[i][0])
            ycord[4].append(datMat[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制样本点
    ax.scatter(xcord[0], ycord[0], s=20, c='b', marker='*', alpha=.5)
    ax.scatter(xcord[1], ycord[1], s=20, c='r', marker='D', alpha=.5)
    ax.scatter(xcord[2], ycord[2], s=20, c='c', marker='>', alpha=.5)
    ax.scatter(xcord[3], ycord[3], s=20, c='k', marker='o', alpha=.5)
    ax.scatter(xcord[4], ycord[4], s=20, c='g', marker='<', alpha=.5)
    # 绘制质心
    ax.scatter(myCentroids[0][0], myCentroids[0][1],
               s=100, c='k', marker='+', alpha=.5)
    ax.scatter(myCentroids[1][0], myCentroids[1][1],
               s=100, c='k', marker='+', alpha=.5)
    ax.scatter(myCentroids[2][0], myCentroids[2][1],
               s=100, c='k', marker='+', alpha=.5)
    ax.scatter(myCentroids[3][0], myCentroids[3][1],
               s=100, c='k', marker='+', alpha=.5)
    ax.scatter(myCentroids[4][0], myCentroids[4][1],
               s=100, c='k', marker='+', alpha=.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()


if __name__ == '__main__':
    plotDataSet('testSet.txt')
