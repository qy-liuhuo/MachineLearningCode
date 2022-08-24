# -*- coding: utf-8 -*-
"""
用途：教学
20220321  sqx
"""

from matplotlib.font_manager import FontProperties
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
    # 计算特征个数，由于最后一列为y值所以减一
    numFeat = len(open(filename).readline().split('\t')) - 1
    xArr = []
    yArr = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr, yArr


"""
函数说明：使用局部加权线性回归计算回归系数w

Parameters:
    testPoint - 测试样本点
    xArr - x数据集
    yArr - y数据集
    k - 高斯核的k，自定义参数
    
Returns:
    testPoint上的回归结果


"""
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    # 创建加权对角阵
    weights = np.mat(np.eye((m)))
    for j in range(m):
        # 高斯核
        diffMat = testPoint - xMat[j, :]
        ######################### 填空1-sqx #############################
        weights[j,j]=np.exp(diffMat*diffMat.T/(-2*pow(k,2)))
        #填空1-sqx  计算 w[j,j] 
        #
        
    
    ######################### 填空2-sqx #############################
    #填空2-sqx  计算 xTx（局部加权后）
    #
    xTx=xMat.T*weights*xMat
    
    
    # 求矩阵的行列式
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵，不能求逆")
        return
    #########################  填空3-sqx #############################
    #填空3-sqx  求系数矩阵ws   提示：     .I  ：求逆矩阵
    #
    ws=xTx.I*xMat.T*weights*yMat
    
    #########################  填空4-sqx #############################
    #返回测试样本点testPoint的预测结果
    #
    return testPoint*ws
    


"""
函数说明：局部加权线性回归测试

Parameters:
    testArr - 测试数据集
    xArr - x数据集
    yArr - y数据集
    k - 高斯核的k,自定义参数
    
Returns:
    yHat - testArr上的回归结果


"""
def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


"""
函数说明：绘制多条局部加权回归曲线

Parameters:
    None
    
Returns:
    None


"""
def plotlwlrRegression():
    # 设置中文字体
    font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=14)
    xArr, yArr = loadDataSet('ex1.txt')
    yHat_1 = lwlrTest(xArr, xArr, yArr, 1.0)
    yHat_2 = lwlrTest(xArr, xArr, yArr, 0.01)
    yHat_3 = lwlrTest(xArr, xArr, yArr, 0.003)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    
    yMat_1 = np.mat(yHat_1)
    yMat_2 = np.mat(yHat_2)
    yMat_3 = np.mat(yHat_3)
    
    
    
    
    ##########################  sqx 计算相关性 ############################
  
    print(np.corrcoef(yHat_1.T, yArr))
    print(np.corrcoef(yHat_2.T, yArr))
    print(np.corrcoef(yHat_3.T, yArr))
    
    # 排序，返回索引值
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:, 0, :]
   
    fig, axs = plt.subplots(nrows=6, ncols=1, sharex=False, sharey=False, figsize=(10, 30))
    axs[0].plot(xSort[:, 1], yHat_1[srtInd], c='red')
    axs[1].plot(xSort[:, 1], yHat_2[srtInd], c='red')
    axs[2].plot(xSort[:, 1], yHat_3[srtInd], c='red')
    
    axs[0].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)
    axs[1].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)
    axs[2].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)
    
    axs[3].scatter(xMat[:, 1].flatten().A[0], (yMat-yMat_1).flatten().A[0], s=20, c='blue', alpha=.5)
    axs[4].scatter(xMat[:, 1].flatten().A[0], (yMat-yMat_2).flatten().A[0], s=20, c='blue', alpha=.5)
    axs[5].scatter(xMat[:, 1].flatten().A[0], (yMat-yMat_3).flatten().A[0], s=20, c='blue', alpha=.5)
    
    axs0_title_text = axs[0].set_title(u'局部加权回归曲线,k=1.0', FontProperties=font)
    axs1_title_text = axs[1].set_title(u'局部加权回归曲线,k=0.01', FontProperties=font)
    axs2_title_text = axs[2].set_title(u'局部加权回归曲线,k=0.003', FontProperties=font)
    
    plt.setp(axs0_title_text, size=8, weight='bold', color='red')
    plt.setp(axs1_title_text, size=8, weight='bold', color='red')
    plt.setp(axs2_title_text, size=8, weight='bold', color='red')
    plt.xlabel('X')
    plt.show()
    
    
if __name__ == '__main__':
    plotlwlrRegression()

