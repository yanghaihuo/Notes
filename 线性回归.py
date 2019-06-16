import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
income = pd.read_csv(r'C:\Users\Lenovo\Desktop\Salary_Data.csv')
# 绘制散点图  工作年限与收入之间的散点图
sns.lmplot(x = 'YearsExperience', y = 'Salary', data = income, ci = None)
# 显示图形
plt.show()


# 简单线性回归模型的参数求解
# 样本量
n = income.shape[0]
# 计算自变量、因变量、自变量平方、自变量与因变量乘积的和
sum_x = income.YearsExperience.sum()
sum_y = income.Salary.sum()
sum_x2 = income.YearsExperience.pow(2).sum()
xy = income.YearsExperience * income.Salary
sum_xy = xy.sum()
# 根据公式计算回归模型的参数
b = (sum_xy-sum_x*sum_y/n)/(sum_x2-sum_x**2/n)
a = income.Salary.mean()-b*income.YearsExperience.mean()
# 打印出计算结果
print('回归参数a的值：',a)
print('回归参数b的值：',b)

import statsmodels.api as sm
# 利用收入数据集，构建回归模型
fit = sm.formula.ols('Salary ~ YearsExperience', data = income).fit()
# 返回模型的参数值
print(fit.params)



# 多元线性回归模型的构建和预测
from sklearn import model_selection
Profit = pd.read_excel(r'C:\Users\Lenovo\Desktop\Predict to Profit.xlsx')
# 将数据集拆分为训练集和测试集
train, test = model_selection.train_test_split(Profit, test_size = 0.2, random_state=1234)
# 根据train数据集建模
model = sm.formula.ols('Profit ~ RD_Spend + Administration + Marketing_Spend + C(State)', data = train).fit()
print('模型的偏回归系数分别为：\n', model.params)
# 删除test数据集中的Profit变量，用剩下的自变量进行预测
test_X = test.drop(labels = 'Profit', axis = 1)
pred = model.predict(exog = test_X)
print('对比预测值和实际值的差异：\n',pd.DataFrame({'Prediction':pred,'Real':test.Profit}))


# 生成由State变量衍生的哑变量
dummies = pd.get_dummies(Profit.State)
print(dummies)
print(type(dummies))

# 将哑变量与原始数据集水平合并
Profit_New = pd.concat([Profit,dummies], axis = 1)
# 删除State变量和New York变量（因为State变量已被分解为哑变量，New York变量需要作为参照组）
Profit_New.drop(labels = ['State','New York'], axis = 1, inplace = True)


# 拆分数据集Profit_New
train, test = model_selection.train_test_split(Profit_New, test_size = 0.2, random_state=1234)
# 建模
model2 = sm.formula.ols('Profit ~ RD_Spend + Administration + Marketing_Spend + Florida + California', data = train).fit()
print('模型的偏回归系数分别为：\n', model2.params)



# 一、回归模型的假设检验
# 1.模型的显著性检验————F检验
import numpy as np
# 计算建模数据中，因变量的均值
ybar = train.Profit.mean()
# 统计变量个数和观测个数
p = model2.df_model
n = train.shape[0]
# 计算回归离差平方和
RSS = np.sum((model2.fittedvalues-ybar) ** 2)
# 计算误差平方和
ESS = np.sum(model2.resid ** 2)
# 计算F统计量的值
F = (RSS/p)/(ESS/(n-p-1))
print('F统计量的值：',F)
# 返回模型中的F值
print(model2.fvalue)

from scipy.stats import f
# 计算F分布的理论值
F_Theroy = f.ppf(q=0.95, dfn = p,dfd = n-p-1)
print('F分布的理论值为：',F_Theroy)

# 对比统计量的值和理论F分布的值，如果计算的统计量值超过理论的值，则拒绝原假设，否则需接受原假设。
# H0:模型的所有偏回归系数全为0；H1:至少有一个变量可以构成因变量的线性组合。
# 就F检验而言，研究者往往是更加希望通过数据来推翻原假设H0,而接受备择假设H1的结论。



# 2.回归系数的显著性检验————t检验
# H0:第j变量的偏回归系数为0，即认为该变量不是因变量的影响因素；H1：第j变量是影响因变量的重要因素。
# 模型的概览信息
print(model2.summary())
# 每个t统计量值都对应了概率值p,用来判断统计量是否显著的直接办法，通常概率值p小于0.05时表示拒绝原假设。
# 从返回的结果可知，只有截距项Intercept和研发成本RD_Spend对应的p值小于0.05，说明其余变量都没有通过系数的显著性检验，
# 即在模型中这些变量不是影响利润的重要因素。


# 二、回归模型的诊断
# 1，误差项服从正态分布(实质是要求因变量服从正态分布)；
# 2，无多重共线性(模型中的自变量之间没有存在较高的相关关系)；
# 3，线性相关性(确保用于建模的自变量和因变量之间存在线性关系)；
# 4，误差项的独立性(实质是对因变量y的独立性检验)；
# 5，方差齐性(要求模型残差项的方差不随自变量的变动而呈现某种趋势，否则残差的趋势就可以被自变量刻画)。



# 1.正态性检验(对原数据集中的利润变量进行正态性检验)
# 1.1直方图法
import scipy.stats as stats
# 绘制直方图
sns.distplot(a = Profit_New.Profit, bins = 10, fit = stats.norm, norm_hist = True,
             hist_kws = {'color':'steelblue', 'edgecolor':'black'}, 
             kde_kws = {'color':'black', 'linestyle':'--', 'label':'核密度曲线'}, 
             fit_kws = {'color':'red', 'linestyle':':', 'label':'正态密度曲线'})
# 显示图例
plt.legend()
# 显示图形
plt.show()


# 1.2残差的正态性检验（PP图和QQ图法）
pp_qq_plot = sm.ProbPlot(Profit_New.Profit)
# 绘制PP图
pp_qq_plot.ppplot(line = '45')
plt.title('P-P图')
# 绘制QQ图
pp_qq_plot.qqplot(line = 'q')
plt.title('Q-Q图')
# 显示图形
plt.show()


import scipy.stats as stats
# 1.3 Shapiro检验和K-S检验(这两种检验方法均属于非参数方法，他们的原假设被设定为变量服从正态分布，两者的最大区别
#    在于适用的数据量不一样，若数据低于5000，则使用Shapiro,否则使用K-S检验法)
print(stats.shapiro(Profit_New.Profit))
# out: (0.979339838,0.53790229558)
# 元组中的第一个元素是Shapiro检验的统计量，第二个元素是对应的概率值p.由于p值大于
# 置信水平0.05，故接受利润变量服从正态分布的原假设。
# 生成正态分布和均匀分布随机数
rnorm = np.random.normal(loc = 5, scale=2, size = 10000)
runif = np.random.uniform(low = 1, high = 100, size = 10000)
# 正态性检验
KS_Test1 = stats.kstest(rvs = rnorm, args = (rnorm.mean(), rnorm.std()), cdf = 'norm')
KS_Test2 = stats.kstest(rvs = runif, args = (runif.mean(), runif.std()), cdf = 'norm')
print(KS_Test1)# 正态分布随机数的检验pvalue > 置信水平0.05，则需接受原假设；
print(KS_Test2)# 均匀分布随机数的检验pvalue远远小于0.05，则需拒绝原假设。


# 2.多重共线性检验(VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
# 自变量X(包含RD_Spend、Marketing_Spend和常数列1)
X = sm.add_constant(Profit_New.ix[:,['RD_Spend','Marketing_Spend']])
print(X)
# 构造空的数据框，用于存储VIF值
vif = pd.DataFrame()
vif["features"] = X.columns
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
# 返回VIF值
print(vif)
# 多重共线性用方差膨胀因子VIF来鉴定:VIF > 10,说明变量间存在多重共线性；VIF > 100，说明变量间存在严重的多重共线性。



# 3.线性相关性检验(Pearson)
# 计算数据集Profit_New中每个自变量与因变量利润之间的相关系数
Profit_New.drop('Profit', axis = 1).corrwith(Profit_New.Profit)
print(Profit_New.drop('Profit', axis = 1).corrwith(Profit_New.Profit))
# 线性相关的程度说明(|p|>=0.8 高度相关；0.5<=|p|<0.8 中度相关；0.3<=|p|<0.5 弱相关；|p|<0.3 几乎不相关)
# 散点图矩阵
# 绘制散点图矩阵
sns.pairplot(Profit_New.ix[:,['RD_Spend','Administration','Marketing_Spend','Profit']])
# 显示图形
plt.show()

# 模型修正
model3 = sm.formula.ols('Profit ~ RD_Spend + Marketing_Spend', data = train).fit()
# 模型回归系数的估计值
print(model3.params)



# 4.异常值检验
outliers = model3.get_influence()

# 4.1高杠杆值点（帽子矩阵）
leverage = outliers.hat_matrix_diag


# 4.2 dffits值
dffits = outliers.dffits[0]
# 4.3学生化残差
resid_stu = outliers.resid_studentized_external


# 4.4 cook距离
cook = outliers.cooks_distance[0]

# 合并各种异常值检验的统计量值
contat1 = pd.concat([pd.Series(leverage, name = 'leverage'),pd.Series(dffits, name = 'dffits'),
                     pd.Series(resid_stu,name = 'resid_stu'),pd.Series(cook, name = 'cook')],axis = 1)
# 重设train数据的行索引
train.index = range(train.shape[0])
# 将上面的统计量与train数据集合并
profit_outliers = pd.concat([train,contat1], axis = 1)
print(profit_outliers)

# 计算异常值数量的比例(当标准化残差大于2时，即认为对应的数据点为异常值)  标准化残差都在 -2~2 之间
outliers_ratio = sum(np.where((np.abs(profit_outliers.resid_stu)>2),1,0))/profit_outliers.shape[0]
print('标准化残差：\n',outliers_ratio)
#out: 0.025
# 如果异常样本的比例不高(<=5%),可以考虑将异常点删除；
# 如果异常样本的比例非常高，选择删除会丢失一些重要信息，所以需要衍生哑变量，即对于异常点，设置哑变量的值为1，否则为0。

# 挑选出非异常的观测点
none_outliers = profit_outliers.ix[np.abs(profit_outliers.resid_stu)<=2,]
# 应用无异常值的数据集重新建模
model4 = sm.formula.ols('Profit ~ RD_Spend + Marketing_Spend', data = none_outliers).fit()
print(model4.params)



# 5.独立性检验(Durbin-Watson)
# Durbin-Watson统计量
# 模型概览
print(model4.summary())

# DW值在2左右，表明残差项之间是不相关的；如果与2偏离的较远，则说明不满足残差的独立性假设。



# 6.方差齐性检验(若模型残差不满足齐性，采用模型变换法或加权最小二乘法)
# 6.1图形法
# 设置第一张子图的位置
ax1 = plt.subplot2grid(shape = (2,1), loc = (0,0))
# 绘制散点图
ax1.scatter(none_outliers.RD_Spend, (model4.resid-model4.resid.mean())/model4.resid.std())
# 添加水平参考线
ax1.hlines(y = 0 ,xmin = none_outliers.RD_Spend.min(),xmax = none_outliers.RD_Spend.max(), color = 'red', linestyles = '--')
# 添加x轴和y轴标签
ax1.set_xlabel('RD_Spend')
ax1.set_ylabel('Std_Residual')

# 设置第二张子图的位置
ax2 = plt.subplot2grid(shape = (2,1), loc = (1,0))
# 绘制散点图
ax2.scatter(none_outliers.Marketing_Spend, (model4.resid-model4.resid.mean())/model4.resid.std())
# 添加水平参考线
ax2.hlines(y = 0 ,xmin = none_outliers.Marketing_Spend.min(),xmax = none_outliers.Marketing_Spend.max(), color = 'red', linestyles = '--')
# 添加x轴和y轴标签
ax2.set_xlabel('Marketing_Spend')
ax2.set_ylabel('Std_Residual')

# 调整子图之间的水平间距和高度间距
plt.subplots_adjust(hspace=0.6, wspace=0.3)
# 显示图形
plt.show()


# 6.2 BP检验(原假设是残差的方差为一个常数，通过构造拉格朗日乘子LM统计量，实现方差齐性的检验)
sm.stats.diagnostic.het_breushpagan(model4.resid, exog_het = model4.model.exog)
# out：(1.46751,0.4801,0.7029,0.5019)
# 第一个值为LM统计量；第二个值是统计量对应的概率p值，该值大于0.05，说明接收残差为常数的原假设；
# 第三个值为F统计量，用于检验残差平方项与自变量之间是否独立，如果独立则表明残差方差齐性；
# 第四个值为F统计量的概率p值，同样大于0.05，则进一步表示残差满足方差齐性的假设。

#6.3回归模型的预测
# 模型预测
# model4对测试集的预测
pred4 = model4.predict(exog = test.ix[:,['RD_Spend','Marketing_Spend']])
# 绘制预测值与实际值的散点图
plt.scatter(x = test.Profit, y = pred4)
# 添加斜率为1，截距项为0的参考线
plt.plot([test.Profit.min(),test.Profit.max()],[test.Profit.min(),test.Profit.max()],
        color = 'red', linestyle = '--')
# 添加轴标签
plt.xlabel('实际值')
plt.ylabel('预测值')
# 显示图形
plt.show()




'''
Α α：阿尔法 Alpha
Β β：贝塔 Beta
Γ γ：伽玛 Gamma
Δ δ：德尔塔 Delte
Ε ε：艾普西龙 Epsilon
Ζ ζ ：捷塔 Zeta
Ε η：依塔 Eta
Θ θ：西塔 Theta
Ι ι：艾欧塔 Iota
Κ κ：喀帕 Kappa
∧ λ：拉姆达 Lambda
Μ μ：缪 Mu
Ν ν：拗 Nu
Ξ ξ：克西 Xi
Ο ο：欧麦克轮 Omicron
∏ π：派 Pi
Ρ ρ：柔 Rho
∑ σ：西格玛 Sigma
Τ τ：套 Tau
Υ υ：宇普西龙 Upsilon
Φ φ：fai Phi
Χ χ：器 Chi
Ω ω：欧米伽 Omega
'''
