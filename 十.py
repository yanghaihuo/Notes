# euro12[euro12.Team.str.startswith('G')]

# euro12.loc[euro12.Team.isin(['England', 'Italy', 'Russia']), ['Team','Shooting Accuracy']]
#
# data.query('month == 1 and day == 2')

# import warnings
# warnings.filterwarnings('ignore')
# sns.pairplot(dat,kind="reg")
# plt.show()

# sales.rename(columns = {'购药时间':'销售时间'}, inplace = True)

# sales['销售时间'], sales['销售星期'] = sales['销售时间'].str.split(' ', 1).str

# 时间戳转换成字符串日期时间   datetime.datetime.fromtimestamp(timestamp[, tz])	根据指定的时间戳创建一个datetime对象
ed['film_date'] = ted['film_date'].apply(lambda x: datetime.datetime.fromtimestamp(int(x)).strftime('%d-%m-%Y'))

import datetime

end_time = 1525104000000
d = datetime.datetime.fromtimestamp(end_time / 1000)  # 时间戳转换成字符串日期时间
str1 = d.strftime("%Y-%m-%d %H:%M:%S.%f")
print(d)  # 2018-05-01 00:00:00
print(str1) # 2018-05-01 00:00:00.000000


# 计算词汇量
df['wc'] = df['transcript'].apply(lambda x: len(x.split()))



import networkx as nx

# 建network
G = nx.Graph()
edges = list(zip(related_df['title'], related_df['related']))
G.add_edges_from(edges)
# 画图
plt.figure(figsize=(25, 25))
nx.draw(G, with_labels=False)


df["GoodWine"] = df.quality.apply(lambda x: 1 if x >=6 else 0)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 训练集与测试集8：2切分
X_train, X_test, y_train, y_test = train_test_split(X.values,y.values, test_size = 0.2, random_state=20)
# 2. 标准化数据
scalerX = StandardScaler() #
scalerX.fit(X_train)       # 使用training data的标准差进行标准化
X_train = scalerX.transform(X_train) # 对训练集特征进行标准化
X_test = scalerX.transform(X_test)   # 对测试机特征进行标准化


# 计数

# count:字符串 列表
# x为字符串或列表
# x.count('熏儿')

# from collections import Counter
# array =  [1, 2, 3, 3, 2, 1, 0, 2]
# def counter(arr):
#     return Counter(arr).most_common(2) # 返回出现频率最高的两个数
# 结果：[(2, 3), (1, 2)]
# 从一个可iterable对象（list、tuple、dict、字符串等）创建

# 列属性为object修改为数值型
# cols = data.columns[data.dtypes.eq(object)]
# data[cols] = data[cols].apply(pd.to_numeric, errors='coerce', axis=0)