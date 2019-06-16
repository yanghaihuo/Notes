'''
下采样：以数据量少的一方的样本数量为准，对于一个不均衡的数据，让目标值(如0和1分类)中的样本数据量相同。
上采样：以数据量多的一方的样本数量为准，把样本数量较少的类的样本数量生成和样本数量多的一方相同。

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
df = data_train.append(data_test) 将训练集和测试集进行合并


Other变量&类别变量：stripplot()和swarmplot()
sns.stripplot(x='day',y='total_bill',data=tips,jitter=True)
sns.swarmplot(x='total_bill',y='day',hue='sex',data=tips)

类别特征对应的特征分布：boxplot()和violinplot()
sns.boxplot(x="day", y="total_bill", hue="weekend", data=tips, dodge=False)
sns.violinplot(x="day", y="total_bill", hue="time", split=True, data=tips)

类别特征的统计信息：barplot() pointplt()和countplot()
sns.barplot(x='sex',y='survived',hue='class',data=titanic)
sns.pointplot(x="sex", y="survived", hue="class", data=titanic,
              palette={"First": "g", "Second": "m", "Third":'b'},
              markers=["^", "o","+"], linestyles=["-", "--",""]);
sns.countplot(hue='sex',x='survived',data=titanic,palette='Greens_d')

多层面板分类图:factorplot()和FacetGrid
sns.factorplot(x="time", y="total_bill", hue="smoker",hue_order=["No","Yes"]
               ,col="day", data=tips, kind="box", size=4, aspect=.5,
              palette="Set3");



条形图barplot
fig,axes=plt.subplots(1,2，,figsize=(20,5))
sns.barplot(x="age",y="color",data=data,ax=axes[0])  #左图
sns.barplot(x="color",y="age",data=data,ax=axes[1])  #右图

点图pointplot
sns.pointplot(x="smoker",y="age",data=data,hue="gender",dodge=True,markers=["*","x"],linestyles=["-.","--"])

计数图countplot:不能同时输入x和y，且countplot没有误差棒
fig,axes=plt.subplots(1,2)
sns.countplot(x="gender",data=data,ax=axes[0]) #左图
sns.countplot(y="gender",data=data,ax=axes[1])  #右图


FacetGrid 是一个绘制多个图表（以网格形式显示）的接口。
可展示三个变量的条件关系，将其中的变量赋值给网格的行和列，并使用不同颜色的绘图元素。
步骤：
1、实例化对象
2、map，映射到具体的 seaborn 图表类型
3、添加图例
# 在不同社会等级下，男性和女性在不同登陆港口下的数量对比
grid = sns.FacetGrid(data_all, col='Pclass', hue='Sex', palette='seismic', size=4)
# 'Embarked' 是  data_all （是一个 DataFrame） 中的字段
grid.map(sns.countplot, 'Embarked', alpha=.8)
# 在图表的右边会显示图例
grid.add_legend()

grid = sns.FacetGrid(data_all, row='Sex', col='Pclass',
                     hue='Survived', palette='seismic', size=4)
grid.map(sns.countplot, 'Embarked', alpha=0.8)
grid.add_legend()

#散点图
g = sns.FacetGrid(df, col="origin")
g.map(plt.scatter, "horsepower", "mpg")

e = sns.FacetGrid(data1, col = 'Embarked')
e.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95.0, palette = 'deep')
e.add_legend()

sns.kdeplot(data_train.loc[(data_train['Survived'] == 0),'Age'] , color='gray',shade=True,label='not survived')
sns.kdeplot(data_train.loc[(data_train['Survived'] == 1),'Age'] , color='g',shade=True, label='survived')





seaborn.factorplot

seaborn.factorplot(x=None, y=None, hue=None, data=None, row=None,
col=None, col_wrap=None, estimator=<function mean>, ci=95, n_boot=1000,
units=None, order=None, hue_order=None, row_order=None, col_order=None,
kind='point', size=4, aspect=1, orient=None, color=None, palette=None,
legend=True, legend_out=True, sharex=True, sharey=True, margin_titles=False, facet_kws=None, **kwargs)

Parameters：

x,y,hue 数据集变量 变量名
date 数据集 数据集名
row,col 更多分类变量进行平铺显示 变量名
col_wrap 每行的最高平铺数 整数
estimator 在每个分类中进行矢量到标量的映射 矢量
ci 置信区间 浮点数或None
n_boot 计算置信区间时使用的引导迭代次数 整数
units 采样单元的标识符，用于执行多级引导和重复测量设计 数据变量或向量数据
order, hue_order 对应排序列表 字符串列表
row_order, col_order 对应排序列表 字符串列表
kind : 可选：point 默认, bar 柱形图, count 频次, box 箱体, violin 提琴, strip 散点，swarm 分散点（具体图形参考文章前部的分类介绍）
size 每个面的高度（英寸） 标量
aspect 纵横比 标量
orient 方向 "v"/"h"
color 颜色 matplotlib颜色
palette 调色板 seaborn颜色色板或字典
legend hue的信息面板 True/False
legend_out 是否扩展图形，并将信息框绘制在中心右边 True/False
share{x,y} 共享轴线 True/False
facet_kws FacetGrid的其他参数 字典



# 箱型图特征分析
fig, [ax1,ax2] = plt.subplots(1,2,figsize=(20,6))
sns.boxplot(x="Pclass", y="Age", data=data_train, ax =ax1)
sns.swarmplot(x="Pclass", y="Age", data=data_train, ax =ax1)
sns.kdeplot(data_train.loc[(data_train['Pclass'] == 3),'Age'] , color='b',shade=True, label='Pcalss3',ax=ax2)
sns.kdeplot(data_train.loc[(data_train['Pclass'] == 1),'Age'] , color='g',shade=True, label='Pclass1',ax=ax2)
sns.kdeplot(data_train.loc[(data_train['Pclass'] == 2),'Age'] , color='r',shade=True, label='Pclass2',ax=ax2)
ax1.set_title('Age特征在Pclass下的箱型图', fontsize = 18)
ax2.set_title("Age特征在Pclass下的kde图", fontsize = 18)
fig.show()


g = sns.pairplot(data_train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked']], hue='Survived', palette = 'seismic',size=4,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=50) )
g.set(xticklabels=[])


用“0”替换Cabin缺失值，并将非缺失Cabin特征值提取出第一个值以进行分类，通过codes量化为数字
df['CabinCat'] = pd.Categorical.from_array(df.Cabin.fillna('0').apply(lambda x: x[0])).codes

df.loc[(df['surname']=='abbott')&(df['Age']==35),'Parch'] = 2
# 从Name中提取Title信息，因为同为男性，Mr.和 Master.的生还率是不一样的
df["Title"] = df["Name"].apply(lambda x: re.search(' ([A-Za-z]+)\.',x).group(1))
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 2, "Mme": 3,"Don": 9,"Dona": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
# 量化Title信息
df["TitleCat"] = df.loc[:,'Title'].map(title_mapping)
# SibSp和Parch特征进行组合
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
# 根据FamilySize分布进行分箱
df["FamilySize"] = pd.cut(df["FamilySize"], bins=[0,1,4,20], labels=[0,1,2])
# 从Name特征衍生出Name的长度
df["NameLength"] = df["Name"].apply(lambda x: len(x))
# 量化Embarked特征
df["Embarked"] = pd.Categorical.from_array(df.Embarked).codes
# 对Sex特征进行独热编码分组
df = pd.concat([df,pd.get_dummies(df['Sex'])],axis=1)









'''
'''
常见的加密方法 RSA、MD5、BASE64、urlcode、urlencode 
import execjs

def exec_js_function(js):
    # 编译JS代码
    ctx = execjs.compile("""
        function getDictLabel(data, value, defaultValue){
            for (var i = 0; i < data.length; i++){
                var row = data[i];
                if (row.value == value){
                    return row.label;
                }
            }
            return defaultValue;
        }
    """)
    # 删除一些无关的字符
    jscode = js.replace('document.write(s);', '').replace(', true);', ')').replace('var s =', '')
    # 执行代码
    # ctx.call("add", 1, 2)
    return ctx.eval(jscode)

# jiaMiPasswd = execjs.compile(open(r"a41.js").read().decode("utf-8")).call('bingo', passwd)
def getEnPwd(modulus, exponent, pwd):
    jsstr = ""
    for filename in os.listdir('./js'):
        f = open('./js/' + filename)
        jsstr += f.read()
        f.close()

    ctx = execjs.compile(jsstr)
    return ctx.call('getenpwd', modulus, exponent, pwd)
    # print(ctx.call('enString', '123456'))
'''





