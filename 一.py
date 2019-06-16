import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats

# 可视化的中文处理
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')
cd = pd.read_csv(r'C:\Users\Lenovo\Desktop\chengduxinfang.csv',encoding='gbk')
cd = cd[cd['states'].isin(['住宅在售','住宅售罄','住宅未开盘','住宅下期待开']) ]
cd.price = cd.price.str.extract('(.*?)元/平')
cd.dropna(subset=['price'],inplace=True)
cd['price'] = cd['price'].astype(int)
# print(cd['price'].unique())
# print(cd.nunique())# 计算DataFrame每一列不重复元素的个数
# print(cd.isnull().sum())# 计算DataFrame每一列缺失值的个数
# print(cd.isnull().sum()/cd.shape[0])
# 计算DataFrame每一列不重复元素的个数
# def func(x):
#     y=len(x.unique())
#     return y
# address_unique=cd.apply(func)
# app_info.comments.describe(percentiles=np.arange(0, 1.2, 0.2))
# print(cd.info())
# print(cd.describe())
# print(cd.describe(include=['object']))
# 新增年龄和工龄两列
# df['age'] = pd.datetime.today().year - df.birthday.dt.year
# df['workage'] = pd.datetime.today().year - df.start_work.dt.year

# loandata['term']=loandata['term'].map(str.strip)
# time.strftime('%Y-%m-%d %H:%M:%S,time.localtime())

# sales['销售时间'], sales['销售星期'] = sales['销售时间'].str.split(' ', 1).str
# sales['销售时间'] = pd.to_datetime(sales['销售时间'],format = '%Y-%m-%d',errors = 'coerce')
# sales = sales[(sales['销售数量'] > 0) & (sales['应收金额'] > 0)]
# month = (sales['销售时间'].max() - sales['销售时间'].min()).days//30

##对去重后的数据按照天进行重新采样
#首先要把索引变成时间
# sales.index = pd.DatetimeIndex(sales['销售时间'])
#然后对其按照每天从新采样
# salesd = sales.resample('D').count()

#画图
# salesd.plot(x = salesd.index, y = '实收金额')
# plt.xlabel('Time')
# plt.ylabel('Money')
# plt.title('xiao shou shu ju')
# plt.show()
#按月采样
# salesm = sales.resample('M').count()



# 数据分列
# grade_split = pd.DataFrame((x.split('-') for x in loandata.grade), index=loandata.index,
#                            columns=['grade', 'sub_grade'])  # 将一列数据拆分开显示为两列
# loandata = pd.merge(loandata, grade_split, how='inner', on="categories", right_index=True,
#                     left_index=True)  # 合并两个pd对象，根据列categories合并
# loandata = pd.merge(loandata, grade_split, right_index=True, left_index=True)  # 直接合并pd对象，两个pd对象顺序一样


sns.set_style("whitegrid")
fig,axes=plt.subplots(1,3) #创建一个一行三列的画布
sns.distplot(cd.price,fit = stats.norm,norm_hist = True,ax=axes[0]) #左图
sns.distplot(cd.price,hist=False,ax=axes[1]) #中图
sns.distplot(cd.price,kde=False,ax=axes[2]) #右图
plt.show()

sns.distplot(cd.price,color='steelblue')
plt.show()

# fig,axes=plt.subplots(1,2)
# sns.distplot(cd.price,norm_hist=True,kde=False,ax=axes[0]) #左图 密度
# sns.distplot(cd.price,kde=False,ax=axes[1]) #右图 计数
# plt.show()




baobao = pd.read_csv(r'C:\Users\Lenovo\Desktop\tianmaotianmaobaobao.csv',encoding='gbk')
baobao=baobao.loc[baobao.sale!='None',:]
baobao.index=range(0,baobao.shape[0])
baobao['new_sale']=baobao.sale.str[:-1].astype('int')
y=lambda x:float(x[:-1]) * 10000 if x.find('万') else float(x)
baobao['new_estimate']=baobao.estimate.apply(y)
baobao=baobao.drop(['sale','estimate'],axis=1)

# data = baobao.corr()
# sns.heatmap(data)
# plt.show()

# print(baobao['new_estimate'].sort_values())


# sns.lmplot(x='price', y='new_estimate',data=baobao,fit_reg=True, scatter=True ) #散点图
# plt.show()
# sns.lmplot(x='price', y='new_estimate',hue='shop',data=baobao,fit_reg=False, scatter=True)
# plt.show()

# train = train[-((train.SalePrice < 200000) &  (train.GrLivArea > 4000))]

# sns.jointplot(x='Area',y='Tprice',data=sh) #散点图
# sns.jointplot(x='Area',y='Tprice',data=sh，kind='hex')
# plt.show()


# lm = sns.lmplot(x = 'Age', y = 'Fare', data = titanic, hue = 'Sex', fit_reg=True)
# lm.set(title = 'Fare x Age')
# axes = lm.axes
# axes[0,0].set_ylim(-5,)
# axes[0,0].set_xlim(-5,85)


# from os import path

# d=path.dirname(__file__)


# df.drop(df[df['职位名称'].str.contains('实习')].index, inplace=True)
# train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

# pattern = '\d+'
# df['work_year'] = df['工作经验'].str.findall(pattern)
# # 数据处理后的工作年限
# avg_work_year = []
# # 工作年限
# for i in df['work_year']:
#    # 如果工作经验为'不限'或'应届毕业生',那么匹配值为空,工作年限为0
#    if len(i) == 0:
#        avg_work_year.append(0)
#    # 如果匹配值为一个数值,那么返回该数值
#    elif len(i) == 1:
#        avg_work_year.append(int(''.join(i)))
#    # 如果匹配值为一个区间,那么取平均值
#    else:
#        num_list = [int(j) for j in i]
#        avg_year = sum(num_list)/2
#        avg_work_year.append(avg_year)
# df['工作经验'] = avg_work_year



# prices_avg = {'0到50/月': 0, '50到100/月': 0, '100到150/月': 0, '150到200/月': 0, '大于200/月': 0}
# 	for d in data:
# 		price = int(float(d.get('price')) / float(d.get('area')))
# 		if price in range(0, 50):
# 			prices_avg['0到50/月'] += 1
# 		elif price in range(50, 100):
# 			prices_avg['50到100/月'] += 1
# 		elif price in range(100, 150):
# 			prices_avg['100到150/月'] += 1
# 		elif price in range(150, 200):
# 			prices_avg['150到200/月'] += 1
# 		else:
# 			prices_avg['大于200/月'] += 1
# 	DrawPie(title='上海租房(单位面积)月租金分布饼图', data=prices_avg, savepath='./results')


# gender_map = {0: 'unknown', 1: 'male', 2: 'female'}
# data['gender'] = data['gender'].apply(lambda x: gender_map[x]) #映射
# data.sample(5) #随机选5行

# newdataset = dataset[-dataset['城区'].isin(['燕郊'])]

# data['year']=data['date'].apply(lambda x:x.year)
# data['month']=data['date'].apply(lambda x:x.month)
# data['day']=data['date'].apply(lambda x:x.day)


# plt.subplot2grid((2,3),(1,0), colspan=2)  # colspan = 2 表示横向跨度是 2
# plt.figure(1, figsize=(40, 60))
plt.subplot(2,2,1)
cd['area'].value_counts().plot(kind='bar')
plt.subplot(2,2,2)
cd['area'].value_counts().plot(kind='hist')
plt.subplot(2,2,3)
cd['area'].value_counts().plot(kind='pie')
plt.subplot(2,2,4)
cd['area'].value_counts().plot(kind='kde')
plt.show()
# plt.subplot(3,2,5)
cd['area'].value_counts().plot(kind='barh')
plt.show()
cd['area'].value_counts().plot(kind='box')
plt.show()
cd['area'].value_counts().plot(kind='area')
plt.show()
cd['area'].value_counts().plot(color='r',marker='D')
plt.show()
baobao.plot(kind='scatter', x='price', y='new_estimate', color='g')
plt.show()
'''Series.plot（kind ='line'，ax = None，figsize = None，use_index = True，title = None，grid = None，legend = False，
  style = None，logx = False，logy = False，loglog = False，xticks = None，yticks = None，xlim = None，ylim = None，rot = None，
  fontsize = None，colormap = None，table = False，yerr = None，xerr = None，label = None，secondary_y = False，** kwds ）'''



import scipy.stats as stats
# 绘制直方图
sns.distplot(a = cd['area'].value_counts(), bins = 10, fit = stats.norm, norm_hist = True,
             hist_kws = {'color':'steelblue', 'edgecolor':'black'},
             kde_kws = {'color':'black', 'linestyle':'--', 'label':'核密度曲线'},
             fit_kws = {'color':'red', 'linestyle':':', 'label':'正态密度曲线'})
# 显示图例
plt.legend()
# 显示图形
plt.show()

# 导入绘图模块
import matplotlib.pyplot as plt
# 设置绘图风格
plt.style.use('ggplot')
# 绘制直方图
sunspots.counts.plot(kind = 'hist', bins = 30, normed = True)
# 绘制核密度图
sunspots.counts.plot(kind = 'kde')
# 图形展现
plt.show()