'''
线性可分SVM 近似线性可分SVM: LinearSVC
非线性可分SVM(核函数)： SVC    kernel:linear(线性核函数) poly(多项式核函数) rbf(高斯核函数 径向基核函数) Sigmoid precomputed(计算一个核矩阵)
线性SVM回归： svm.LinearSVR
非线性SVM回归：svm.SVR
惩罚系数C 核函数 Y值(gamma) epsilon值
根据经验，核函数选择为高斯核函数时模型拟合效果往往会更好：惩罚系数C可以选择在0.0001~10000,值越大，惩罚力度越大，
模型越有可能产生过拟合；高斯核函数中的Y参数越大，对应的支持向量则越少，反之支持向量越多、模型越复杂、越有可能过拟合

SVM优点：由于SVM模型最终所形成的分类器仅依赖于一些支持向量，这就导致模型具有很好的鲁棒性(增加或删除非支持向量的样本点，
        并不会改变分类器的效果)以及避免“维度灾难”的发生(模型并不会随数据维度的提升而提高计算的复杂度)：
        模型具有很好的泛化能力，一定程度上可以避免模型的过拟合：
        也可以避免模型在运算过程中出现的局部最优。
SVM缺点：模型不适合大样本的分类或预测，因为它会消耗大量的计算资源和时间；
        模型对缺失样本非常敏感，这就需要建模前清洗好每一个观测样本；
        虽然可以通过核函数解决非线性可分问题，但是模型对核函数的选择也同样敏感；
        SVM为黑盒模型(相比于回归或预测数等算法)，对计算得到的结果无法解释

np.loglp计算加一后的对数，其逆运算是np.expm1
数据平滑处理： https://blog.csdn.net/qq_36523839/article/details/82422865
回归中对于连续型因变量需要做探索性分析 ，如果数据呈现严重的偏态，需要进行变换(y=np.log1p(forestfires.area))
'''

from sklearn import svm
import pandas as pd
from sklearn import model_selection
from sklearn import metrics

# 读取外部数据
letters = pd.read_csv(r'C:\Users\Lenovo\Desktop\letterdata.csv')
# 数据前5行
print(letters.head())

# 将数据拆分为训练集和测试集
predictors = letters.columns[1:]
X_train,X_test,y_train,y_test = model_selection.train_test_split(letters[predictors], letters.letter, 
                                                                 test_size = 0.25, random_state = 1234)
																 
# 使用网格搜索法，选择线性可分SVM“类”中的最佳C值
C=[0.05,0.1,0.5,1,2,5]
parameters = {'C':C}
grid_linear_svc = model_selection.GridSearchCV(estimator = svm.LinearSVC(),param_grid =parameters,scoring='accuracy',cv=5,verbose =1)
# 模型在训练数据集上的拟合
grid_linear_svc.fit(X_train,y_train)
# 返回交叉验证后的最佳参数值
print(grid_linear_svc.best_params_, grid_linear_svc.best_score_)

# 模型在测试集上的预测
pred_linear_svc = grid_linear_svc.predict(X_test)
# 模型的预测准确率
metrics.accuracy_score(y_test, pred_linear_svc)


# 使用网格搜索法，选择非线性SVM“类”中的最佳C值
kernel=['rbf','linear','poly','sigmoid']
C=[0.1,0.5,1,2,5]
parameters = {'kernel':kernel,'C':C}
grid_svc = model_selection.GridSearchCV(estimator = svm.SVC(),param_grid =parameters,scoring='accuracy',cv=5,verbose =1)
# 模型在训练数据集上的拟合
grid_svc.fit(X_train,y_train)
# 返回交叉验证后的最佳参数值
print(grid_svc.best_params_, grid_svc.best_score_)


# 模型在测试集上的预测
pred_svc = grid_svc.predict(X_test)
# 模型的预测准确率
metrics.accuracy_score(y_test,pred_svc)


# 读取外部数据
forestfires = pd.read_csv(r'C:\Users\Lenovo\Desktop\forestfires.csv')
# 数据前5行
print(forestfires.head())

# 删除day变量
forestfires.drop('day',axis = 1, inplace = True)
# 将月份作数值化处理
forestfires.month = pd.factorize(forestfires.month)[0]
# 预览数据前5行
print(forestfires.head())


import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
# 回归中对于连续型因变量需要做探索性分析 ，如果数据呈现严重的偏态，需要进行变换(y=np.log1p(forestfires.area))
# 绘制森林烧毁面积的直方图
sns.distplot(forestfires.area, bins = 50, kde = True, fit = norm, hist_kws = {'color':'steelblue'}, 
             kde_kws = {'color':'red', 'label':'Kernel Density'}, 
             fit_kws = {'color':'black','label':'Nomal', 'linestyle':'--'})
# 显示图例
plt.legend()
# 显示图形
plt.show()


from sklearn import preprocessing
import numpy as np
from sklearn import neighbors
# 对area变量作对数变换
y = np.log1p(forestfires.area)  # https://blog.csdn.net/qq_36523839/article/details/82422865
# 将X变量作标准化处理
predictors = forestfires.columns[:-1]
X = preprocessing.scale(forestfires[predictors])

# 将数据拆分为训练集和测试集
X_train,X_test,y_train,y_test = model_selection.train_test_split(X, y, test_size = 0.25, random_state = 1234)

# 构建默认参数的SVM回归模型
svr = svm.SVR()
# 模型在训练数据集上的拟合
svr.fit(X_train,y_train)
# 模型在测试上的预测
pred_svr = svr.predict(X_test)
# 计算模型的MSE
metrics.mean_squared_error(y_test,pred_svr)


# 使用网格搜索法，选择SVM回归中的最佳C值、epsilon值和gamma值
epsilon = np.arange(0.1,1.5,0.2)
C= np.arange(100,1000,200)
gamma = np.arange(0.001,0.01,0.002)
parameters = {'epsilon':epsilon,'C':C,'gamma':gamma}
grid_svr = model_selection.GridSearchCV(estimator = svm.SVR(),param_grid =parameters,
                                        scoring='neg_mean_squared_error',cv=5,verbose =1, n_jobs=2)
# 模型在训练数据集上的拟合
grid_svr.fit(X_train,y_train)
# 返回交叉验证后的最佳参数值
print(grid_svr.best_params_, grid_svr.best_score_)


# 模型在测试集上的预测
pred_grid_svr = grid_svr.predict(X_test)
# 计算模型在测试集上的MSE值
print(metrics.mean_squared_error(y_test,pred_grid_svr))
														 