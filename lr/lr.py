# -*- coding:utf-8 -*-
"""
Module Description: 
Date: 2019/10/28 
Author: Wang P
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import matplotlib

# 显示中文标签
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# 字体格式
matplotlib.rcParams['font.family'] = 'sans-serif'
# 正常显示负号
matplotlib.rcParams['axes.unicode_minus'] = False

# from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib.ticker as mtick


def loadDataSet(filename):
    """
    加载数据源
    """
    X = []
    Y = []
    with open(filename, 'rb') as f:
        for idx, line in enumerate(f):
            line = line.decode('utf-8').strip()
            if not line:
                continue

            eles = line.split()
            eles = list(map(float, eles))

            X.append(eles[:-1])
            Y.append([eles[-1]])
    return np.array(X), np.array(Y)


# 假设函数
def h(theta, X):
    """
    假设函数 h(x) = 
    """
    return np.dot(X, theta)  # 矩阵乘法


# 代价函数
def J(theta, X, Y):
    m = len(X)
    return np.sum(np.dot((h(theta, X) - Y).T, (h(theta, X) - Y)) / (2 * m))


# 梯度下降更新公式
def bgd(alpha, maxloop, epsilon, X, Y):
    m, n = X.shape  # m是样本数，n是特征数(包括了全部是1的x0)，其实也就是参数theta的个数

    theta = np.zeros((n, 1))  # 参数theta全部初始化为0

    count = 0  # 记录迭代轮次
    converged = False  # 是否已经收敛的标志
    error = np.inf  # 当前的代价函数值
    errors = [J(theta, X, Y), ]  # 记录每一次迭代得代价函数值

    thetas = {}
    for i in range(n):
        thetas[i] = [theta[i, 0], ]  # 记录每一个theta j的历史更新

    while count <= maxloop:
        if (converged):
            break
        count = count + 1

        # 这里，我们的梯度计算统一了
        for j in range(n):
            deriv = np.dot(X[:, j].T, (h(theta, X) - Y)).sum() / m
            thetas[j].append(theta[j, 0] - alpha * deriv)

        for j in range(n):
            theta[j, 0] = thetas[j][-1]

        error = J(theta, X, Y)
        errors.append(error)

        if (abs(errors[-1] - errors[-2]) < epsilon):
            converged = True
    return theta, errors, thetas


def standarize(X):
    """特征标准化处理

    Args:
        X 样本集
    Returns:
        标准后的样本集
    """
    m, n = X.shape
    # 归一化每一个特征
    for j in range(n):
        features = X[:, j]
        meanVal = features.mean(axis=0)
        std = features.std(axis=0)
        if std != 0:
            X[:, j] = (features - meanVal) / std
        else:
            X[:, j] = 0
    return X


if __name__ == '__main__':
    data_set = os.path.dirname(os.path.abspath(__file__))+'/houses.txt'
    ori_X, Y = loadDataSet(data_set)  # 数据
    print(ori_X.shape)
    print(Y.shape)

    m, n = ori_X.shape  # 获取样本数和特征数
    # 数据标准化
    X = standarize(ori_X.copy())
    X = np.concatenate((np.ones((m, 1)), X), axis=1)  # 数组拼接函数

    alpha = 1  # 学习率
    maxloop = 5000  # 最大迭代次数
    epsilon = 0.000001  # 收敛判断条件

    theta, errors, thetas = bgd(alpha, maxloop, epsilon, X, Y)

    # 预测
    normalizedSize = (70 - ori_X[:, 0].mean(0)) / ori_X[:, 0].std(0)
    normalizedBr = (2 - ori_X[:, 1].mean(0)) / ori_X[:, 1].std(0)
    predicateX = np.matrix([[1, normalizedSize, normalizedBr]])
    price = h(theta, predicateX)
    print('70㎡两居估价: ￥%.4f万元' % price)

    # %matplotlib
    # 打印拟合平面
    fittingFig = plt.figure(figsize=(16, 12))
    title = 'bgd: rate=%.3f, maxloop=%d, epsilon=%.3f \n' % (alpha, maxloop, epsilon)
    ax = fittingFig.gca(projection='3d')

    xx = np.linspace(0, 200, 25)
    yy = np.linspace(0, 5, 25)
    zz = np.zeros((25, 25))
    for i in range(25):
        for j in range(25):
            normalizedSize = (xx[i] - ori_X[:, 0].mean(0)) / ori_X[:, 0].std(0)
            x = np.matrix([[1, normalizedSize, normalizedBr]])
            zz[i, j] = h(theta, x)
    xx, yy = np.meshgrid(xx, yy)
    ax.zaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap=cm.rainbow, alpha=0.1, antialiased=True)

    xs = ori_X[:, 0].flatten()
    ys = ori_X[:, 1].flatten()
    zs = Y[:, 0].flatten()
    ax.scatter(xs, ys, zs, c='b', marker='o')

    ax.set_xlabel(u'面积')
    ax.set_ylabel(u'卧室数')
    ax.set_zlabel(u'估价')

    plt.show()

    errorsFig = plt.figure()
    ax = errorsFig.add_subplot(111)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

    ax.plot(range(len(errors)), errors)
    ax.set_xlabel(u'迭代次数')
    ax.set_ylabel(u'代价函数')
    plt.show()
