# 逻辑回归假设数据服从伯努利分布,通过极大化似然函数的方法，运用梯度下降来求解参数，来达到将数据二分类的目的。
# AUC>=0.8  k-s>=0.4,越大越好

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 自定义绘制ks曲线的函数
def plot_ks(y_test, y_score, positive_flag):
    # 对y_test,y_score重新设置索引
    y_test.index = np.arange(len(y_test))
    # y_score.index = np.arange(len(y_score))
    # 构建目标数据集
    target_data = pd.DataFrame({'y_test':y_test, 'y_score':y_score})
    # 按y_score降序排列
    target_data.sort_values(by = 'y_score', ascending = False, inplace = True)
    # 自定义分位点
    cuts = np.arange(0.1,1,0.1)
    # 计算各分位点对应的Score值
    index = len(y_score)*cuts
    scores = y_score[index.astype('int')]
    # 根据不同的Score值，计算Sensitivity和Specificity
    Sensitivity = []
    Specificity = []
    for score in scores:
        # 正例覆盖样本数量与实际正例样本量
        positive_recall = target_data.loc[(target_data.y_test == positive_flag) & (target_data.y_score>score),:].shape[0]
        positive = sum(target_data.y_test == positive_flag)
        # 负例覆盖样本数量与实际负例样本量
        negative_recall = target_data.loc[(target_data.y_test != positive_flag) & (target_data.y_score<=score),:].shape[0]
        negative = sum(target_data.y_test != positive_flag)
        Sensitivity.append(positive_recall/positive)
        Specificity.append(negative_recall/negative)
    # 构建绘图数据
    plot_data = pd.DataFrame({'cuts':cuts,'y1':1-np.array(Specificity),'y2':np.array(Sensitivity), 
                              'ks':np.array(Sensitivity)-(1-np.array(Specificity))})
    # 寻找Sensitivity和1-Specificity之差的最大值索引
    max_ks_index = np.argmax(plot_data.ks)
    plt.plot([0]+cuts.tolist()+[1], [0]+plot_data.y1.tolist()+[1], label = '1-Specificity')
    plt.plot([0]+cuts.tolist()+[1], [0]+plot_data.y2.tolist()+[1], label = 'Sensitivity')
    # 添加参考线
    plt.vlines(plot_data.cuts[max_ks_index], ymin = plot_data.y1[max_ks_index], 
               ymax = plot_data.y2[max_ks_index], linestyles = '--')
    # 添加文本信息
    plt.text(x = plot_data.cuts[max_ks_index]+0.01,
             y = plot_data.y1[max_ks_index]+plot_data.ks[max_ks_index]/2,
             s = 'KS= %.2f' %plot_data.ks[max_ks_index])
    # 显示图例
    plt.legend()
    # 显示图形
    plt.show()

# 导入虚拟数据
virtual_data = pd.read_excel(r'C:\Users\Lenovo\Desktop\virtual_data.xlsx')
# 应用自定义函数绘制k-s曲线
plot_ks(y_test = virtual_data.Class, y_score = virtual_data.Score,positive_flag = 'P')	




from sklearn import linear_model,model_selection

# 读取数据
sports = pd.read_csv(r'C:\Users\Lenovo\Desktop\Run or Walk.csv')
# 提取出所有自变量名称
predictors = sports.columns[4:]
# 构建自变量矩阵
X = sports.ix[:,predictors]
# 提取y变量值
y = sports.activity
# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.25, random_state = 1234)

# 利用训练集建模
sklearn_logistic = linear_model.LogisticRegression()
sklearn_logistic.fit(X_train, y_train)
# 返回模型的各个参数
print(sklearn_logistic.intercept_, sklearn_logistic.coef_)


# 模型预测
sklearn_predict = sklearn_logistic.predict(X_test)
print(sklearn_predict)
# 预测结果统计
pd.Series(sklearn_predict).value_counts()
print(pd.Series(sklearn_predict).value_counts())




from sklearn import metrics
# 混淆矩阵
cm = metrics.confusion_matrix(y_test, sklearn_predict, labels = [0,1])
print(cm)


Accuracy = metrics.scorer.accuracy_score(y_test, sklearn_predict)
Sensitivity = metrics.scorer.recall_score(y_test, sklearn_predict)
Specificity = metrics.scorer.recall_score(y_test, sklearn_predict, pos_label=0)
print('模型准确率为%.2f%%:' %(Accuracy*100)) # 分类准确率（即所有分类中被正确分类的比例) (TP + TN)/(TP + TN + FN + FN)
print('正例覆盖率为%.2f%%' %(Sensitivity*100)) #召回率：正确识别的正例个数在实际为正例的样本数中的占比 Recall = TP/(TP + FN) 样本
print('负例覆盖率为%.2f%%' %(Specificity*100))
metrics.precision_score(y_test,sklearn_predict)# 精确率：预测为真的正样本占所有预测为正样本的比例 Precision = TP/(TP + FP) 预测


# 混淆矩阵的可视化
import seaborn as sns
# 绘制热力图
sns.heatmap(cm, annot = True, fmt = '.2e',cmap = 'GnBu')
# 图形显示
plt.show()

'''
ROC曲线是真正率和假正率在不同的阀值之间的图形表示关系。通常用作权衡模型的敏感度与模型对一个错误分类报警的概率。
真正率（TPR）：表示正的样本被预测为正占所有正样本的比例。
假正率（FPR）：表示负的样本被预测为正占所有负样本的比例。
计算不同阈值下，fpr和tpr的组合值，其中fpr表示1-Specificity，tpr表示Sensitivity
真正率（TPR） = 灵敏度 = 召回率 = TP/(TP+FN)
假正率（FPR） = 1- 特异度 = FP/(FP+TN)
thresholds = 阈值
'''

# ROC曲线与AUC值
# 注意：此实现仅限于二进制分类任务
# y得分为模型预测正例的概率    y_score表示每个测试样本属于正样本的概率。
y_score = sklearn_logistic.predict_proba(X_test)[:,1]
print(sklearn_logistic.predict_proba(X_test))

fpr,tpr,threshold = metrics.roc_curve(y_test, y_score)
# 计算AUC的值
roc_auc = metrics.auc(fpr,tpr)
# 绘制面积图
# plt.stackplot(fpr, tpr, color='steelblue', alpha = 0.5, edgecolor = 'black')
# 添加边际线
plt.plot(fpr, tpr, color='black', lw = 1)
# 添加对角线
plt.plot([0,1],[0,1], color = 'red', linestyle = '--')
# 添加文本信息
plt.text(0.5,0.3,'ROC curve (area = %0.2f)' % roc_auc)
# 添加x轴与y轴标签
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
# 显示图形
plt.show()


# 调用自定义函数，绘制K-S曲线
plot_ks(y_test = y_test, y_score = y_score, positive_flag = 1)




# -----------------------第一步 建模 ----------------------- #
import statsmodels.api as sm
# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.25, random_state = 1234)
# 为训练集和测试集的X矩阵添加常数列1
X_train2 = sm.add_constant(X_train)
X_test2 = sm.add_constant(X_test)
# 拟合Logistic模型
sm_logistic = sm.formula.Logit(y_train, X_train2).fit()
# sklearn_logistic = linear_model.LogisticRegression()
# sklearn_logistic.fit(X_train, y_train)
# 返回模型的参数
print(sm_logistic.params)


# -----------------------第二步 预测构建混淆矩阵 ----------------------- #
# 模型在测试集上的预测
sm_y_probability = sm_logistic.predict(X_test2)
print(sm_y_probability)
# 根据概率值，将观测进行分类，以0.5作为阈值
sm_pred_y = np.where(sm_y_probability >= 0.5, 1, 0)
# 混淆矩阵
cm = metrics.confusion_matrix(y_test, sm_pred_y, labels = [0,1])
print(cm)


# -----------------------第三步 绘制ROC曲线 ----------------------- #
# 计算真正率和假正率
fpr,tpr,threshold = metrics.roc_curve(y_test, sm_y_probability)
# 计算auc的值
roc_auc = metrics.auc(fpr,tpr)
# 绘制面积图
# plt.stackplot(fpr, tpr, color='steelblue', alpha = 0.5, edgecolor = 'black')
# 添加边际线
plt.plot(fpr, tpr, color='black', lw = 1)
# 添加对角线
plt.plot([0,1],[0,1], color = 'red', linestyle = '--')
# 添加文本信息
plt.text(0.5,0.3,'ROC curve (area = %0.2f)' % roc_auc)
# 添加x轴与y轴标签
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
# 显示图形
plt.show()


# -----------------------第四步 绘制K-S曲线 ----------------------- #
# 调用自定义函数，绘制K-S曲线
sm_y_probability.index = np.arange(len(sm_y_probability))
plot_ks(y_test = y_test, y_score = sm_y_probability, positive_flag = 1)

'''

1.为什么可以用似然函数？
目标是要让预测为正的的概率最大，且预测为负的概率也最大，
即每一个样本预测都要得到最大的概率，将所有的样本预测后的概率进行相乘都最大，
然后取对数，再乘以负的m分之一，就得到了损失函数，这就用到了似然函数。

2.逻辑回归为什么一般性能差？
LR是线性的，不能得到非线性关系，实际问题并不完全能用线性关系就能拟合。

3.使用L1L2正则化，为什么可以降低模型的复杂度？
模型越复杂，越容易过拟合，L1正则化给了模型的拉普拉斯先验，L2正则化给了模型的高斯先验。
从参数的角度来看，L1得到稀疏解，去掉一部分特征降低模型复杂度。
L2得到较小的参数，如果参数很大，样本稍微变动一点，值就有很大偏差，相当于降低每个特征的权重。

4.那么为什么L1能得到稀疏解呢？
L1正则化是L1范数而来，投到坐标图里面，是棱型的，最优解在坐标轴上取到，所以某些部分的特征的系数就为0。

5.L1正则化不可导，怎么求解？
坐标轴下降法（按照每个坐标轴一个个使其收敛），最小角回归（是一个逐步的过程，每一步都选择一个相关性很大的特征，总的运算步数只和特征的数目有关，和训练集的大小无关）


'''

