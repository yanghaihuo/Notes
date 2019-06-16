'''
高偏差：欠拟合  高方差：过拟合
偏差指的是模型预测值与真实值的差异，是由使用的学习算法的某些错误或过于简单的假设造成的误差。它会导致模型欠拟合，很难有高的预测准确率。
方差指的是不同训练数据训练的模型的预测值之间的差异，它是由于使用的算法模型过于复杂，导致对训练数据的变化十分敏感，这样会导致模型过拟合，使得模型带入了过多的噪音。

from sklearn.preprocessing import StandardScaler

#标准化，返回值为标准化后的数据
StandardScaler().fit_transform(iris.data)

from sklearn.preprocessing import MinMaxScaler

#区间缩放，返回值为缩放到[0, 1]区间的数据
MinMaxScaler().fit_transform(iris.data)

from sklearn.preprocessing import Normalizer

#归一化，返回值为归一化后的数据
Normalizer().fit_transform(iris.data)

标准化是依照特征矩阵的列处理数据，其通过求z-score的方法，将样本的特征值转换到同一量纲下。
归一化是依照特征矩阵的行处理数据，其目的在于样本向量在点乘运算或其他核函数计算相似性时，拥有统一的标准，也就是说都转化为“单位向量”。





直接得到原始数据的对应的序号列表，将类别信息转化成数值信息应用到模型中去
pd.Categorical( list ).codes

mushrooms[column] = pd.factorize(mushrooms[column])[0] # 返回两个值，取第一个
直接得到原始数据的对应的序号列表，将类别信息转化成数值信息应用到模型中去
mushrooms[column] =pd.Categorical(mushrooms[column]).codes


df['CabinCat'] = pd.Categorical.from_array(df.Cabin.fillna('0').apply(lambda x: x[0])).codes    # pd.Categorical.from_array  numpy数组接口
fig, ax = plt.subplots(figsize=(10,5))
sns.countplot(x='CabinCat', hue='Survived',data=df)
plt.show()



# 妇女/儿童 男士标签
child_age = 18
def get_person(passenger):
    age, sex = passenger
    if (age < child_age):
        return 'child'
    elif (sex == 'female'):
        return 'female_adult'
    else:
        return 'male_adult'

df = pd.concat([df, pd.DataFrame(df[['Age', 'Sex']].apply(get_person, axis=1), columns=['person'])],axis=1)
df = pd.concat([df,pd.get_dummies(df['person'])],axis=1)

numpy.ravel() vs numpy.flatten()
两者所要实现的功能是一致的（将多维数组降位一维），两者的区别在于返回拷贝（copy）还是返回视图（view），
numpy.flatten()返回一份拷贝，对拷贝所做的修改不会影响（reflects）原始矩阵，
而numpy.ravel()返回的是视图（view，也颇有几分C/C++引用reference的意味），会影响（reflects）原始矩阵。


经典部分回顾：https://mp.weixin.qq.com/s/gAqxE9MA8VPHzaysYLSjsQ
利用随机森林预测填补缺失值
Age特征缺失值：Age有20%缺失值，缺失值较多，大量删除会减少样本信息，由于它与Cabin不同，这里将利用其它特征进行预测填补Age，也就是拟合未知Age特征值。
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor

classers = ['Fare','Parch','Pclass','SibSp','TitleCat',
            'CabinCat','female','male', 'Embarked', 'FamilySize', 'NameLength','Ticket_Numbers','Ticket_Id']
etr = ExtraTreesRegressor(n_estimators=200,random_state=0)
X_train = df[classers][df['Age'].notnull()]
Y_train = df['Age'][df['Age'].notnull()]
X_test = df[classers][df['Age'].isnull()]

etr.fit(X_train.as_matrix(),np.ravel(Y_train))
age_preds = etr.predict(X_test.as_matrix())
df['Age'][df['Age'].isnull()] = age_preds


方差分析   X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
单变量特征选取  返回k个最佳特征
单变量特征提取的原理是分别计算每个特征的某个统计指标，根据该指标来选取特征。
SelectKBest、SelectPercentile，前者选择排名前k个的特征，后者选择排名在前k%的特征。选择的统计指标需要指定，对于regression问题，使用f_regression指标;对于classification问题，可以使用chi2或者f_classif指标。
from sklearn.feature_selection import SelectKBest,chi2

X_new=SelectKBest(chi2,k=2).fit_transform(test_X,test_Y)
False Positive Rate，假阳性率
chi2,卡方统计量，X中特征取值必须非负。卡方检验用来测度随机变量之间的依赖关系。通过卡方检验得到的特征之间是最可能独立的随机变量，因此这些特征的区分度很高。

ANOVA方差分析的 F值 来对各个特征变量打分，打分的意义是：各个特征变量对目标变量的影响权重
from sklearn.feature_selection import SelectKBest, f_classif,chi2

target = data_train["Survived"].values
features= ['female','male','Age','male_adult','female_adult', 'child','TitleCat',
           'Pclass','Ticket_Id','NameLength','CabinType','CabinCat', 'SibSp', 'Parch',
           'Fare','Embarked','Surname_Numbers','Ticket_Numbers','FamilySize',
           'Ticket_dead_women','Ticket_surviving_men',
           'Surname_dead_women','Surname_surviving_men']

train = df[0:891].copy()
test = df[891:].copy()

selector = SelectKBest(f_classif, k=len(features))
selector.fit(train[features], target)
scores = -np.log10(selector.pvalues_)
indices = np.argsort(scores)[::-1]
print("Features importance :")
for f in range(len(scores)):
    print("%0.2f %s" % (scores[indices[f]],features[indices[f]]))










特征选择
单变量特征选择
>>> from sklearn.datasets import load_iris
>>> from sklearn.feature_selection import SelectKBest
>>> from sklearn.feature_selection import chi2
>>> iris = load_iris()
>>> X, y = iris.data, iris.target
>>> X.shape
(150, 4)
>>> X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
>>> X_new.shape
(150, 2)

iris = load_iris()
# X, y = iris.data, iris.target
# X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
# print(X_new)
model = SelectKBest(chi2, k=2)#选择k个最佳特征
model.fit_transform(iris.data, iris.target)#iris.data是特征数据，iris.target是标签数据，该函数可以选择出k个特征
print(model.scores_) # 得分
print(model.pvalues_) # p-values


基于树的特征选择
>>> from sklearn.ensemble import ExtraTreesClassifier
>>> from sklearn.datasets import load_iris
>>> iris = load_iris()
>>> X, y = iris.data, iris.target
>>> X.shape
(150, 4)
>>> clf = ExtraTreesClassifier()
>>> X_new = clf.fit(X, y).transform(X)
>>> clf.feature_importances_  
array([ 0.04...,  0.05...,  0.4...,  0.4...])
>>> X_new.shape               
(150, 2)




上采样：
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print('训练集的交易记录条数：',X_train.shape[0])
print('测试集的交易记录条数：',X_test.shape[0])
print('交易记录总数：',X_train.shape[0] + X_test.shape[0])
print('上采样前，类别为‘1’的共有{}个，类别为‘0’的共有{}个。'.format(sum(y_train==1),sum(y_train==0)))
print('------------------------')

# 对训练集进行上采样处理
smote = SMOTE(random_state=2)
X_train_os,y_train_os = smote.fit_sample(X_train, y_train.ravel()) # ravel(): change the shape of y to (n_samples, )

print('上采样后，训练集的交易记录条数：', len(X_train_os))
print('其中，训练集X的shape:',X_train_os.shape,'，y的shape:',y_train_os.shape)
print('交易记录总数：',X_train_os.shape[0] + X_test.shape[0])
print('上采样后，类别为‘1’的共有{}个，类别为‘0’的共有{}个。'.format(sum(y_train_os==1),sum(y_train_os==0)))

在正负样本都非常之少的情况下，应该采用数据合成的方式；SMOTE
在负样本足够多，正样本非常之少且比例及其悬殊的情况下，应该考虑一分类方法；
在正负样本都足够多且比例不是特别悬殊的情况下，应该考虑采样或者加权的方法。


# 3倍标准差定义异常值
ageMean = np.mean(data_train['age'])
ageStd = np.std(data_train['age'])
ageUpLimit = round((ageMean + 3*ageStd),2)
ageDownLimit = round((ageMean - 3*ageStd),2)
print('年龄异常值上限为：{0}, 下限为：{1}'.format(ageUpLimit,ageDownLimit))

# 四分位距观察异常值
agePercentile = np.percentile(data_train['age'],[0,25,50,75,100])
ageIQR = agePercentile[3] - agePercentile[1]
ageUpLimit = agePercentile[3]+ageIQR*1.5
ageDownLimit = agePercentile[1]-ageIQR*1.5
print('年龄异常值上限为：{0}, 下限为：{1}'.format(ageUpLimit,ageDownLimit))
print('上届异常值占比：{0} %'.format(data_train[data_train['age']>96].shape[0]*100/data_train.shape[0]))
print('下届异常值占比：{0} %'.format(data_train[data_train['age']<8].shape[0]*100/data_train.shape[0]))

data_train.loc[(data_train['Num30-59late']>=8), 'Num30-59late'] = 8
Num30_59lateDlq = data_train.groupby(['Num30-59late'])['IsDlq'].sum()
Num30_59lateAll = data_train.groupby(['Num30-59late'])['IsDlq'].count()
Num30_59lateGroup = Num30_59lateDlq/Num30_59lateAll
Num30_59lateGroup.plot(kind='bar',figsize=(10,5))

Num30_59lateDf = pd.DataFrame(Num30_59lateDlq)
Num30_59lateDf['All'] = Num30_59lateAll
Num30_59lateDf['BadRate'] = Num30_59lateGroup
Num30_59lateDf











df = pd.read_csv("haiwang.csv",sep=",",header=None,names=["nickName","cityName","content","approve","reply","startTime","avatarurl","score"])
data["content_length"] = data["content"].apply(len)


df = pd.read_csv('douban.csv', header=None, names=["quote", "score", "info", "title", "people"])
(dom1, dom2, dom3, dom4) = ([], [], [], [])
# 清洗数据,获取电影年份及国家,增加年份列及国家列
for i in df['info']:
    country = i.split('/')[1].split(' ')[0].strip()
    if country in ['中国大陆', '台湾', '香港']:
        dom1.append(1)
    else:
        dom1.append(0)
    dom2.append(int(i.split('/')[0].replace('(中国大陆)', '').strip()))
df['country'] = dom1
df['year'] = dom2
# 清洗数据,建立评价人数列
for i in df['people']:
    dom3.append(int(i.replace('人评价', '')))
df['people_num'] = dom3
# 生成电影排名列表
dom4 = [x for x in range(1, 251)]
df['rank'] = dom4

# corr()方法:计算两两相关的列,不包括NA/Null值 persion:标准相关系数
print(df[['rank', 'score']].corr(method='pearson')


# 生成带辅助线的散点图矩阵,hue:分类
sns.pairplot(df[['score', 'people_num', 'year', 'country', 'rank']], hue='country', kind='reg', diag_kind='kde', size=1.5)

# distplot:集合功能,kde:显示核密度估计图,fit:控制拟合的参数分布图形,本次为拟合正态分布
sns.distplot(df.score, kde=True, fit=stats.norm)

ctime = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(info['ctime']))

data['time'] = data['ctime'].apply(lambda x: x[:10])
time = data[['time']].copy()
time['time_comment'] = 1
time = time.groupby(by=['time']).count()

data["semiscore"] = data['comment'].apply(lambda x: SnowNLP(x).sentiments)
data['semilabel'] = data["semiscore"].apply(lambda x: 1 if x>0.5 else -1)




#词云图
import jieba
comment=''.join(data['comment'])
wordlist = jieba.cut(comment, cut_all=False)
stopwords_chinese = [line.strip() for line in open('stopwords_chinese.txt',encoding='UTF-8').readlines()]
#过滤掉单个字
word_list=[]
for seg in wordlist:
    if seg not in stopwords_chinese:
        word_list.append(seg)

word_list=pd.DataFrame({'comment':word_list})
word_rank = word_list["comment"].value_counts()

from pyecharts import WordCloud
wordcloud_chinese = WordCloud(width=1500, height=820)
wordcloud_chinese.add("", word_rank.index[0:100], word_rank.values[0:100], word_size_range=[20, 200], is_more_utils=True)
wordcloud_chinese.render("comment.html")










图表布局 Grid
两图结合 Overlap

df['salary'] = df.salary.map({"low": 0, "medium": 1, "high": 2})

批量梯度下降（BGD）、随机梯度下降（SGD）、小批量随机梯度下降（MSGD）
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='warn',
          n_jobs=None, penalty='l2', random_state=None, solver='warn',
          tol=0.0001, verbose=0, warm_start=False)
参数C是正则化项参数的倒数, C的数值越小, 惩罚的力度越大. penalty可选L1, L2正则化项, 默认是L2正则化.
参数solver可选{‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}这5个优化算法:
newton-cg, lbfgs是拟牛顿法, liblinear是坐标轴下降法, sag, saga是随机梯度下降法,
saga可以适用于L1和L2正则化项, 而sag只能用于L2正则化项.

SMOTE是一种过采样算法，它构造新的小类样本而不是产生小类中已有的样本的副本。它基于距离度量选择小类别下两个或者更多的相似样本，
然后选择其中一个样本，并随机选择一定数量的邻居样本对选择的那个样本的一个属性增加噪声，每次处理一个属性。这样就构造了许多新数据。




field = data.price
Q1 = np.quantile(data[field], 0.25)
Q3 = np.quantile(data[field], 0.75)
deta = (Q3 - Q1) * 1.5
data = data[(data[field] >= Q1 - deta) & (data[field] <= Q3 + deta)]

k-means:
随机生成K个聚类中心，
内循环同时进行簇分配：看样本哪些离各自的簇中心近，并进行分配
移动聚类中心：算出各自簇中样本的均值并把簇中心移动到该处，重复进行簇分配和移动聚类中心，直到迭代结束

随机生成K个聚类中心，内循环同时进行簇分配和移动聚类中心

1  随机选取K个点, 作为初始的K个聚类中心
2  计算每个样本点到K个聚类中心的距离, 并将其分给距离最短的簇
3  计算K个簇中所有样本点的均值, 将这K个均值作为K个新的聚类中心
4  重复第2步和第3步, 直到聚类中心不再改变时停止算法, 输出聚类结果

优点:
1 原理简单, 计算速度快
2 聚类效果较理想.
缺点:
1 K值以及初始质心对结果影响较大, 且不好把握.
2 在大数据集上收敛较慢.
3 由于目标函数(簇内离差平方和最小)是非凸函数, 因此通过迭代只能保证局部最优.
4 对于离群点较敏感, 这是由于其是基于均值计算的, 而均值易受离群点的影响.
5 由于其是基于距离进行计算的, 因此通常只能发现"类圆形"的聚类.
注意:
1 对数据异常值的处理；
2 对数据标准化处理（x-min(x))/(max(x)-min(x)）；
3 每一个类别的数量要大体均等；（
4 不同类别间的特质值应该差异较大


import matplotlib.pyplot as plt

K = range(1, 10)
sse = []
for k in K:
    km = KMeans(n_clusters=k, random_state=10)
    km.fit(del_df)
    sse.append(km.inertia_)
plt.figure(figsize=(8, 6))
plt.plot(K, sse, '-o', alpha=0.7)
plt.xlabel("K")
plt.ylabel("簇内误差平方和(SSE)")
plt.show()

inertias：是K-Means模型对象的属性，它作为没有真实分类结果标签下的非监督式评估指标。
          表示样本到最近的聚类中心的距离总和。值越小越好，越小表示样本在类间的分布越集中。

平行坐标图
from pandas.plotting import parallel_coordinates
#训练模型
km = KMeans(n_clusters=2, random_state=10)
km.fit(del_df)
centers = km.cluster_centers_
labels =  km.labels_
customer = pd.DataFrame({'0': centers[0], "1": centers[1]}).T
customer.columns = del_df.keys()
df_median = pd.DataFrame({'2': del_df.median()}).T
customer = pd.concat([customer, df_median])
customer["category"] = ["customer_1", "customer_2", 'median']
#绘制图像
plt.figure(figsize=(12, 6))
parallel_coordinates(customer, "category", colormap='flag'')
plt.xticks(rotation = 15)
plt.show()


#将聚类后的标签加入数据集
del_df['category'] = labels
del_df['category'] = np.where(del_df.category == 0, 'customer_1', 'customer_2')
customer = pd.DataFrame({'0': centers[0], "1": centers[1]}).T
customer["category"] = ['customer_1_center', "customer_2_center"]
customer.columns = del_df.keys()
del_df = pd.concat([del_df, customer])
#对6类产品每年消费水平进行绘制图像
df_new = del_df[['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen', 'category']]
plt.figure(figsize=(18, 6))
parallel_coordinates(df_new, "category", colormap='cool')
plt.xticks(rotation = 15)
plt.show()



信息增益的弊端：对可取值数目较多的属性有所偏好。因为信息增益反映的是给定一个条件以后不确定性减少的程度，
               必然是分得越细的数据集确定性更高，也就是条件熵越小，信息增益越大。

1 不能处理连续特征
2 用信息增益作为标准容易偏向于取值较多的特征
3 不能处理缺失值
4 容易发生过拟合问题


信息增益和信息增益率选择最大值，基尼系数选择最小


偏差(欠拟合)是模型所做的简化假设，使得目标函数更加容易求解
方差(过拟合)是在给定不同训练数据集的情况下，目标函数估计值所改变的量


Z-score：分类、聚类中使用距离来度量相似性或者使用PCA技术进行降维
min-max：在不涉及距离度量、协方差计算、数据不符合正太分布的时。比如图像处理中，将RGB图像转换为灰度图像后将其值限定在[0 255]的范围。

'''

