"""
用途：教学
20220314  sqx
"""
import matplotlib.pyplot as plt
import numpy as np

"""
函数说明：加载数据

Parameters:
    filename - 文件名
    
Returns:
    xArr - x数据集
    yArr - y数据集

"""
def loadDataSet(filename):
    
    
    xArr = []
    yArr = []
    fr = open(filename)
    # 计算特征个数，由于最后一列为y值所以减1
    numFeat = len(fr.readline().split('\t')) - 1

    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr, yArr


"""
函数说明：计算回归系数w

Parameters:
    xArr - x数据集
    yArr - y数据集
    
Returns:
    ws - 回归系数

"""
def standRegres(xArr, yArr):
    xMat = np.mat(xArr)
    #########################  填空1-sqx #############################
    #填空1-sqx  学习xMat生成方式，生成yMat。  yMat为yArr转成矩阵后再转置的结果。
    #
    
    ######################### 填空2-sqx #############################
    #填空2-sqx  计算 xTx
    #
    
    # 求矩阵的行列式
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵，不能求逆")
        return
    #########################  填空3-sqx #############################
    #填空3-sqx  求系数矩阵ws   提示：     .I  ：求逆矩阵
    #
    
    return ws


"""
函数说明：绘制数据集以及回归直线

"""
def plotDataSet():
    xArr, yArr = loadDataSet('ex0.txt')
    ws = standRegres(xArr, yArr)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    xCopy = xMat.copy()
    # 排序
    xCopy.sort(0)
    yHat = xCopy * ws     
    
    ##########################  填空4-sqx ############################
    #填空4-sqx  用plot绘制出 回归出来的直线，用红色
    #
    
    # 绘制样本点即
    # flatten返回一个折叠成一维的数组。但是该函数只能适用于numpy对象，即array或者mat，普通的list列表是不行的
    # 矩阵.A(等效于矩阵.getA())变成了数组
    plt.scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()    
    
if __name__ == '__main__':
    plotDataSet()
    