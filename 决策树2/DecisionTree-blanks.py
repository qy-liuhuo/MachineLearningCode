# -*- coding: utf-8 -*-
"""

用途：教学
202204016  sqx
任务：预测隐形眼镜类型：不可佩戴、硬性、软性
数据的Labels依次是age、prescript、astigmatic、tearRate、class
年龄、症状、是否散光、眼泪数量、分类标签
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np
import pydotplus
# scikit-learn环境版本为1.0.2 在该版本中已经删除six，可直接php install six 然后引入
from six import StringIO
from sklearn import tree

if __name__ == '__main__':
    # 加载文件
    with open('lenses.txt') as fr:
        #######################  填空-sqx  1 #############################
        # 填空-sqx  为下面一行代码添加注释
        # strip默认会删除掉字符串开头和结尾的空格，所以对每行先删除空格并用制表符分割字符串
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # 提取每组数据的类别，保存在列表里
    lenses_targt = []
    for each in lenses:
        # 存储Label(最后一个)到lenses_targt中
        lenses_targt.append([each[-1]])
    # 特征标签
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # 保存lenses数据的临时列表
    lenses_list = []
    # 保存lenses数据的字典，用于生成pandas
    lenses_dict = {}
    # 提取信息，生成字典
    for each_label in lensesLabels:
        for each in lenses:
            # index方法用于从列表中找出某个值第一个匹配项的索引位置
            lenses_list.append(each[lensesLabels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    # 打印字典信息
    # print(lenses_dict)
    # 生成pandas.DataFrame用于对象的创建
    lenses_pd = pd.DataFrame(lenses_dict)
    # 打印数据
    print(lenses_pd)
    # 创建LabelEncoder对象
    le = LabelEncoder()
    # 为每一列序列化
    for col in lenses_pd.columns:
        # fit_transform()干了两件事：fit找到数据转换规则，并将数据标准化
        # transform()直接把转换规则拿来用,需要先进行fit
        # transform函数是一定可以替换为fit_transform函数的，fit_transform函数不能替换为transform函数
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    #######################  填空-sqx  2 #############################
    # 填空-sqx  为下面一行代码添加注释
    # 输出序列化结果，序列化，简单来说就是给文本内容编号
    print(lenses_pd)
    # 创建DecisionTreeClassifier()类
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=4)
    #######################  填空-sqx  3 #############################
    # 填空-sqx  为下面一行代码添加注释
    # 用决策树模型进行拟合，第一个 参数为数据，第二个为label结果

    print(lenses_pd.values.tolist())
    print(lenses_targt)
    clf = clf.fit(lenses_pd.values.tolist(), lenses_targt)
    # 输出
    dot_data = StringIO()
    # 绘制决策树
    tree.export_graphviz(clf, out_file=dot_data, feature_names=lenses_pd.keys(),
                         class_names=clf.classes_, filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # 保存绘制好的决策树，以PDF的形式存储。
    graph.write_pdf("tree.pdf")
    #######################  填空-sqx  4 #############################
    # 填空-sqx  为下面一行代码添加注释
    # 预测1,1,1，0 这种情况下的结果
    print(clf.predict([[1, 1, 1, 0]]))
