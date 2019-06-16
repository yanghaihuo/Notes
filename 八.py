# 字段的拆分 split(pat=None, n = -1, expand=True)  True表示在不同的列展开, 否则以序列的形式显示.
df_new = df["Surname_Age"].str.split("_")

# df_new = df["Surname_Age"].str.split("_").rename(columns={0: "Surname", 1: "Age"})
# df_new.columns = ["Surname","Age"]

# df_mer[df_mer.Age.between(23,28)]

# 字符匹配: df[df.字段名.str.contains("字符", case = True, na =False)]
# contains()函数中case=True表示区分大小写, 默认为True; na = False表示不匹配缺失值.
# df[df.SpouseAge.str.contains("2")]

# df[pd.isnull(df.字段名)]表示匹配该字段中有缺失值的记录.
# df[pd.isnull(df.SpouseAge)]

# pd.cut(x, bins, right = True, labels = None)  right = True表示右边闭合, 左边不闭合

# df2[df2['A'].map(lambda x:x.startswith('61'))]  # 筛选出以61开头的数据

# pd.to_datetime(arg, format = None)
# df["BirthDate"].apply(lambda x: datetime.strptime(x, "%Y/%m/%d")) 转化为日期格式
# df["BD"].apply(lambda x: datetime.strftime(x, "%d-%m-%Y %H:%M:%S")) 转化为字符串


#对species分组求均值
# df_gbsp = df.groupby("species", as_index = False).mean()


# 堆积柱形图
# df_gbsp.plot(kind = "barh", stacked = True)

# 累加直方图
# df_gbsp.plot(kind = "hist", cumulative='True', bins = 20)

# 累积频率直方图
# plt.hist(y,
#          bins = np.arange(min_price,max_price+1000,1000),
#          normed = True, # 设置为频率直方图
#          cumulative = True, # 积累直方图
#          color = 'steelblue', # 指定填充色
#          )


# start_urls = ['https://www.guokr.com/ask/{0}/?page={1}'.format(str(m),str(n)) for m in urls for n in range(1, 101)]


#生成Pclass_Survived的列联表
# Pclass_Survived = pd.crosstab(df_train['Pclass'], df_train['Survived'])
# Pclass_Survived.plot(kind = 'bar', stacked = True)

# pd.crosstab(df_train.Appellation, df_train.Sex).T
# df_train['Appellation'] = df_train['Appellation'].replace(['Mlle','Ms'], 'Miss')

# df_train['Appellation'] = df_train.Name.apply(lambda x: re.search('\w+\.', x).group()).str.replace('.', '')


# 头衔和幸存相关吗? 绘制柱形图
# Appellation_Survived = pd.crosstab(df_train['Appellation'], df_train['Survived'])
# Appellation_Survived.plot(kind = 'bar')
# plt.xticks(np.arange(len(Appellation_Survived.index)), Appellation_Survived.index, rotation = 360)
# plt.title('Survived status by Appellation')


#生成列联表
Sex_Survived = pd.crosstab(df_train['Sex'], df_train['Survived'])
Survived_len = len(Sex_Survived.count())
Sex_index = np.arange(len(Sex_Survived.index))
single_width = 0.35
for i in range(Survived_len):
    SurvivedName = Sex_Survived.columns[i]
    SexCount = Sex_Survived[SurvivedName]
    SexLocation = Sex_index * 1.05 + (i - 1/2)*single_width
   #绘制柱形图
    plt.bar(SexLocation, SexCount, width = single_width)
    for x, y in zip(SexLocation, SexCount):
        #添加数据标签
        plt.text(x, y, '%.0f'%y, ha='center', va='bottom')
index = Sex_index * 1.05
plt.xticks(index, Sex_Survived.index, rotation=360)
plt.title('Survived status by sex')

# df_train['GroupTicket'] = np.where(df_train.Ticket.isin(Ticket_Count_0), 0, 1)
# train = df_train.copy()
# train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])



# 缺失值填充：采用与头衔相对应的年龄中位数进行填补
#求出每个头衔对应的年龄中位数
Age_Appellation_median = train.groupby('Appellation')['Age'].median()
#在当前表设置Appellation为索引
train.set_index('Appellation', inplace = True)
#在当前表填充缺失值
train.Age.fillna(Age_Appellation_median, inplace = True)
#重置索引
train.reset_index(inplace = True)



#对Age进行分组: 2**10>891分成10组, 组距为(最大值80-最小值0)/10 =8取9
bins = [0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90]
train['GroupAge'] = pd.cut(train.Age, bins)
GroupAge_Survived = pd.crosstab(train['GroupAge'], train['Survived'])
GroupAge_Survived.plot(kind = 'bar')
plt.title('Survived status by GroupAge')


train['Appellation'] = train.Appellation.map({'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Rare': 4})

df_Iris['Species']= df_Iris.Species.apply(lambda x: x.split('-')[1])


# 散点图
sns.lmplot(x='GrLivArea', y='SalePrice', data=train, fit_reg=False, scatter=True)
sns.relplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', style='Species', data=df_Iris )
plt.title('SepalLengthCm and SepalWidthCm data by Species')
# 散点图加直方图
sns.jointplot(x='SepalLengthCm', y='SepalWidthCm', data=df_Iris)
# 质量分布图
sns.distplot(df_Iris.PetalWidthCm, bins=5, hist=True, kde=True)
# 箱线图
sns.boxplot(x='Attribute', y='Data',hue='Species', data=Iris)
# 琴图(箱线图与核密度图)
sns.violinplot(x='Attribute', y='Data', hue='Species', data=Iris )
# 多变量图
sns.pairplot(df_Iris.drop('Id', axis=1), hue='Species',kind="reg")


zhaopin = zhaopin[zhaopin.JobTitle.str.contains('.*?数据.*?|.*?分析.*?|.*?Data.*?')]
zhaopin['Workplace'] = zhaopin.Workplace.str.split('-', expand=True)[0]


# 将薪资Salary分为最高薪资和最低薪资, 另外了解到薪资中单位有元/小时, 元/天, 万/月, 万/年, 千/月, 统一将其转化为千/月
import re
#将5种单元进行编号
zhaopin['Standard'] = np.where(zhaopin.Salary.str.contains('元.*?小时'), 0,
                               np.where(zhaopin.Salary.str.contains('元.*?天'), 1,
                                        np.where(zhaopin.Salary.str.contains('千.*?月'), 2,
                                                 np.where(zhaopin.Salary.str.contains('万.*?月'), 3,
                                                          4))))
#用'-'将Salary分割为LowSalary和HighSalary
SalarySplit = zhaopin.Salary.str.split('-', expand = True)
zhaopin['LowSalary'], zhaopin['HighSalary'] = SalarySplit[0], SalarySplit[1]
#Salary中包含'以上', '以下'或者两者都不包含的进行编号
zhaopin['HighOrLow'] = np.where(zhaopin.LowSalary.str.contains('以.*?下'), 0,
                                np.where(zhaopin.LowSalary.str.contains('以.*?上'), 2,
                                         1))
#匹配LowSalary中的数字, 并转为浮点型
Lower = zhaopin.LowSalary.apply(lambda x: re.search('(\d+\.?\d*)', x).group(1)).astype(float)
#对LowSalary中HighOrLow为1的部分进行单位换算, 全部转为'千/月'
zhaopin.LowSalary = np.where(((zhaopin.Standard==0)&(zhaopin.HighOrLow==1)), Lower*8*21/1000,
                             np.where(((zhaopin.Standard==1)&(zhaopin.HighOrLow==1)), Lower*21/1000,
                                      np.where(((zhaopin.Standard==2)&(zhaopin.HighOrLow==1)), Lower,
                                               np.where(((zhaopin.Standard==3)&(zhaopin.HighOrLow==1)), Lower*10,
                                                        np.where(((zhaopin.Standard==4)&(zhaopin.HighOrLow==1)), Lower/12*10,
                                                                 Lower)))))

#对HighSalary中的缺失值进行填充, 可以有效避免匹配出错.
zhaopin.HighSalary.fillna('0千/月', inplace =True)
#匹配HighSalary中的数字, 并转为浮点型
Higher = zhaopin.HighSalary.apply(lambda x: re.search('(\d+\.?\d*).*?', str(x)).group(1)).astype(float)
#对HighSalary中HighOrLow为1的部分完成单位换算, 全部转为'千/月'
zhaopin.HighSalary = np.where(((zhaopin.Standard==0)&(zhaopin.HighOrLow==1)),zhaopin.LowSalary/21*26,
                              np.where(((zhaopin.Standard==1)&(zhaopin.HighOrLow==1)),zhaopin.LowSalary/21*26,
                                       np.where(((zhaopin.Standard==2)&(zhaopin.HighOrLow==1)), Higher,
                                                np.where(((zhaopin.Standard==3)&(zhaopin.HighOrLow==1)), Higher*10,
                                                         np.where(((zhaopin.Standard==4)&(zhaopin.HighOrLow==1)), Higher/12*10,
                                                                  np.where(zhaopin.HighOrLow==0, zhaopin.LowSalary,
                                                                           zhaopin.LowSalary))))))
#查看当HighOrLow为0时, Standard都有哪些, 输出为2, 4
zhaopin[zhaopin.HighOrLow==0].Standard.unique()
#完成HighOrLow为0时的单位换算
zhaopin.loc[(zhaopin.HighOrLow==0)&(zhaopin.Standard==2), 'LowSalary'] = zhaopin[(zhaopin.HighOrLow==0)&(zhaopin.Standard==2)].HighSalary.apply(lambda x: 0.8*x)
zhaopin.loc[(zhaopin.HighOrLow==0)&(zhaopin.Standard==4), 'HighSalary'] = zhaopin[(zhaopin.HighOrLow==0)&(zhaopin.Standard==4)].HighSalary.apply(lambda x: x/12*10)
zhaopin.loc[(zhaopin.HighOrLow==0)&(zhaopin.Standard==4), 'LowSalary'] = zhaopin[(zhaopin.HighOrLow==0)&(zhaopin.Standard==4)].HighSalary.apply(lambda x: 0.8*x)
#查看当HighOrLow为2时, Srandard有哪些, 输出为4
zhaopin[zhaopin.HighOrLow==2].Standard.unique()
#完成HighOrLow为2时的单位换算
zhaopin.loc[zhaopin.HighOrLow==2, 'LowSalary']  = zhaopin[zhaopin.HighOrLow==2].HighSalary.apply(lambda x: x/12*10)
zhaopin.loc[zhaopin.HighOrLow==2, 'HighSalary'] = zhaopin[zhaopin.HighOrLow==2].LowSalary.apply(lambda x: 1.2*x)
zhaopin.LowSalary , zhaopin.HighSalary = zhaopin.LowSalary.apply(lambda x: '%.1f'%x), zhaopin.HighSalary.apply(lambda x: '%.1f'%x)


from pyecharts import Geo
from collections import Counter
#统计各地区出现次数, 并转换为元组的形式
data = Counter(place).most_common()
#生成地理坐标图
geo =Geo("数据分析岗位各地区需求量", title_color="#fff", title_pos="center", width=1200, height=600, background_color='#404a59')
attr, value =geo.cast(data)
#添加数据点
geo.add('', attr, value, visual_range=[0, 100],visual_text_color='#fff', symbol_size=5, is_visualmap=True, is_piecewise=True)
geo.show_config()
geo.render()


# 将有薪资范围的数据提取出来
other_salary = jobs.salary[~jobs.salary.isin(['薪资面议','校招','1K以下'])]
# 提取出薪资的下界
salary_low = other_salary.str.replace('K','').str.split('-').apply(lambda x: float(x[0]))
# 提取出薪资的上界
salary_high = other_salary.str.replace('K','').str.split('-').apply(lambda x: float(x[1]))




