# df.corr(method='spearman')

# fig, ax = plt.subplots(2, 1, figsize=(12, 6))
# #绘制气温和季节箱线图
# sns.boxplot(x='season', y='temp',data=df, ax=ax[0])
# sns.boxplot(x='group_season', y='temp',data=df, ax=ax[1])
#
# df['group_season'] = np.where((df.month <=5) & (df.month >=3), 1,
#                         np.where((df.month <=8) & (df.month >=6), 2,
#                                  np.where((df.month <=11) & (df.month >=9), 3, 4)))


# import seaborn as sns
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
# #绘制箱线图
# sns.boxplot(x="windspeed", data=df,ax=axes[0][0])
# sns.boxplot(x='casual', data=df, ax=axes[0][1])
# sns.boxplot(x='registered', data=df, ax=axes[1][0])
# sns.boxplot(x='count', data=df, ax=axes[1][1])
# plt.show()


# data['price'].corr(data["x"])
# correlation = train_df.corr()['TARGET'].sort_values()
# print('最正相关的是：\n',correlation.tail(10))
# print('最负相关的是：\n',correlation.head(10))
#
# https://blog.csdn.net/maymay_/article/details/80253068
# features,test_features = features.align(test_features,join='inner',axis=1)
#
# train_df.dtypes.value_counts()
#
# train_df.select_dtypes('object')
#
# model.coef_ 查看训练好模型的参数
#
# for idx, row in df.iterrows():
#     data = jieba.cut(row['content'])
#     data = dict(Counter(data))

# data_max.plot.barh(x='app_name',y='install_count',color=colorline)
# for y, x in enumerate(list((data_max['install_count']))):
#     plt.text(x + 0.1, y - 0.08, '%s' %round(x, 1), ha = 'center', color = colors)
'''


from datetime import datetime
import time
import pytz
# datetime.fromtimestamp 时间戳转换成字符串日期时间
today = datetime.fromtimestamp(int(time.time()), pytz.timezone('Asia/Shanghai')).strftime('%Y_%m_%d')
to_day = datetime.fromtimestamp(int(time.time()))
print(today)
print(to_day)

plt.subplot(2, 1, 1)
plt.subplot(211)



'''