import matplotlib.pyplot as plt
import numpy as np
from pyecharts import Geo
import jieba
import jieba.analyse
from wordcloud import WordCloud,ImageColorGenerator
import seaborn as sns
import pandas as pd
import scipy.stats as stats
import os
import pymysql
from sklearn.cluster import KMeans


os.chdir(r'C:\Users\Lenovo\Desktop\yurongfu')

# 可视化的中文处理
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')


# #连接数据库
# conn = pymysql.connect(host='localhost', user='root', password='Y1413010130H', port=3306, db='cd_hotel', charset='utf8')
# sql = 'select * from cd'
# #读取数据
# df = pd.read_sql(sql, conn)


df = pd.read_csv(r'C:\Users\Lenovo\Desktop\taobao\yurongfu.csv')


# print(df.head())
# print(df.tail())
# print(df.shape)
# print(any(df.duplicated()))
# print(any(df.isnull()))
# print(df.nunique())
# print(df.isnull().sum())
# print(df.notnull().sum())
df.drop_duplicates(['title','userId'],inplace=True)
df.commentCount[df.commentCount==" "]=df.sold[df.commentCount==" "]
# df.commentCount.fillna(0,inplace=True)
# df['comment']=df.comment.str[1:-1].astype('float')
# print(df.describe())
# print(df.describe(include=['object']))
# df.info()
# print(df.shape)
df.reset_index(drop=True,inplace=True)


df.commentCount=df.commentCount.astype('float64')
df.shipping=df.shipping.astype('object')
df['price']=df['originalPrice'] + df['shipping']
df['discount']=df['originalPrice'] - df['priceWap']
df.info()
# 总共店铺
print(len(df.userId.drop_duplicates()))
print(len(df.nick.drop_duplicates()))

import warnings
warnings.filterwarnings("ignore")
# sns.pairplot(df,palette="Set2",kind="reg")
# # sns.pairplot(df, hue='质量', palette = 'seismic',size=4,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=50) )
# plt.savefig('变量关系.png', dpi=600)
# plt.show()



# 连续变量
for i in df.columns[5:8]:
    sns.distplot(a = df[i], fit = stats.norm, norm_hist = True,
                 hist_kws = {'color':'steelblue', 'edgecolor':'black'},
                 kde_kws = {'color':'black', 'linestyle':'--', 'label':'核密度曲线'},
                 fit_kws = {'color':'red', 'linestyle':':', 'label':'正态密度曲线'})
    # sns.distplot(df[i], hist=True, kde=False)
    plt.savefig('13.png', dpi=600)
    plt.show()

sns.distplot(a = df.sold[df.sold.notnull()], fit = stats.norm, norm_hist = True,
             hist_kws = {'color':'steelblue', 'edgecolor':'black'},
             kde_kws = {'color':'black', 'linestyle':'--', 'label':'核密度曲线'},
             fit_kws = {'color':'red', 'linestyle':':', 'label':'正态密度曲线'})
plt.savefig('14.png', dpi=600)
plt.show()



# df['userLevel'].plot(kind='hist',orientation='horizontal', cumulative=True)
df['priceWap'].plot(kind='hist',normed = True,cumulative=True,color='steelblue')
# 添加水平参考线
plt.axhline(y = 0.1, color = 'yellow',linestyle = '--', linewidth = 2)
plt.axhline(y = 0.3, color = 'blue', linestyle = '--', linewidth = 2)
plt.axhline(y = 0.5, color = 'red', linestyle = '--', linewidth = 2)
plt.axhline(y = 0.7, color = 'green', linestyle = '--', linewidth = 2)
plt.title('价格频率分布')
plt.savefig('价格频率分布图.png', dpi=600)
plt.show()



# 分类变量
df.shipping.value_counts(ascending=True).plot(kind='barh',rot=0)
plt.savefig('15.png', dpi=600)
plt.show()

# 堆积柱形图
# df_gbsp.plot(kind = "barh", stacked = True)

df1=df.location.value_counts()
sns.barplot(df1.index,df1.values)
plt.xticks(np.arange(len(df1.index)),df1.index, rotation = 90)
# for x,y in enumerate(price):
#     plt.text(y+0.1,x,'%s' %y,va='center')
for x, y in zip(np.arange(len(df1.index)),df1.values):
    plt.text(x, y, '%.0f' % y, ha='center', va='bottom')
# plt.legend(loc='upper right')
plt.savefig('16.png', dpi=600)
plt.show()


df2=df.zkType.value_counts()
sns.barplot(df2.index,df2.values)
plt.xticks(np.arange(len(df2.index)),df2.index, rotation = 90)
for x, y in zip(np.arange(len(df2.index)),df2.values):
    plt.text(x, y, '%.0f' % y, ha='center', va='bottom')
plt.savefig('17.png', dpi=600)
plt.show()


df['location1']=df.location.replace(regex={r'.*河北.*':'河北','.*广东.*':'广东','.*江苏.*':'江苏','.*香港.*':'香港','.*山东.*':'山东','.*浙江.*':'浙江','.*湖北.*':'湖北','.*湖南.*':'湖南','.*福建.*':'福建','.*安徽.*':'安徽','.*江西.*':'江西','.*吉林.*':'吉林','.*陕西.*':'陕西','.*甘肃.*':'甘肃','.*河南.*':'河南','.*辽宁.*':'辽宁','.*四川.*':'四川','.*广西.*':'广西','.*山西.*':'山西','.*云南.*':'云南'})
df.location1=df.location1[~df.location1.isin(['美国','法国','意大利','日本','加拿大','中国',' ','香港','韩国'])]
df3=df.location1.value_counts()
sns.barplot(df3.index,df3.values)
plt.xticks(np.arange(len(df3.index)),df3.index, rotation = 90)
for x, y in zip(np.arange(len(df3.index)),df3.values):
    plt.text(x, y, '%.0f' % y, ha='center', va='bottom')
# plt.legend(loc='upper right')
plt.savefig('18.png', dpi=600)
plt.show()


data =df.location1.value_counts()
geo =Geo("全国羽绒服数量分布", title_color="#fff", title_pos="center", width=1200, height=600, background_color='#404a59')
geo.add('', data.index, data.values, visual_range=[0, 600],visual_text_color='#fff', symbol_size=5, is_visualmap=True, is_piecewise=True)
geo.render("全国羽绒服数量分布.html")


plt.plot(df.originalPrice,label='原价')
plt.plot(df.price,label='真实价格')
plt.plot(df.discount,label='折扣价格')
plt.legend()
plt.savefig('趋势.png', dpi=600)
plt.show()

# 散点图

# baobao.plot(kind='scatter', x='price', y='new_estimate', color='g')

plt.scatter(df.price,df.sold)
plt.savefig('1.png', dpi=600)
plt.show()

sns.lmplot(x='price', y='sold',hue='location1',data=df,fit_reg=False, scatter=True)
plt.savefig('2.png', dpi=600)
plt.show()
sns.relplot(x='price', y='sold', hue='location1', data=df )
plt.savefig('3.png', dpi=600)
plt.show()

# X轴分类变量散点图
sns.stripplot(x="location1", y="sold", data=df, alpha=".25")
plt.savefig('4.png', dpi=600)
plt.show()
sns.swarmplot(x="location1", y="sold", data=df)
plt.savefig('5.png', dpi=600)
plt.show()


sns.barplot(x="location1", y="commentCount", data=df)
plt.savefig('6.png', dpi=600)
plt.show()

df4=df.shipping.groupby(df['shipping']).count()
plt.bar(df4.index,df4.values)
plt.savefig('7.png', dpi=600)
plt.show()


sns.boxplot(x='location1',y='priceWap',data=df)
sns.stripplot(x="location1", y="priceWap", data=df)
plt.savefig('8.png', dpi=600)
plt.show()


sns.lmplot(x='commentCount', y='sold',hue='location1',data=df,fit_reg=False, scatter=True)
plt.savefig('9.png', dpi=600)
plt.show()



df.sold=df.sold.fillna(df.sold.mean())

from sklearn import preprocessing
# 数据标准化处理
X = preprocessing.minmax_scale(df[['commentCount','price','sold']])
# 将数组转换为数据框
X = pd.DataFrame(X, columns=['评论','价格','销量'])

# 将球员数据集聚为3类
kmeans = KMeans(n_clusters = 3)
kmeans.fit(X)
# 将聚类结果标签插入到数据集players中
X['cluster'] = kmeans.labels_
# 构建空列表，用于存储三个簇的簇中心
centers = []
for i in X.cluster.unique():
    centers.append(X.ix[X.cluster == i,['评论','价格','销量']].mean())
# 将列表转换为数组，便于后面的索引取数
centers = np.array(centers)
print(centers)
sns.lmplot(x = '评论', y = '价格', hue = 'cluster', data = X, markers = ['^','s','o'],
           fit_reg = False, scatter_kws = {'alpha':0.8}, legend = False)
plt.xlabel('评论')
plt.ylabel('价格')
plt.savefig('10.png', dpi=600)
plt.show()

sns.lmplot(x = '评论', y = '销量', hue = 'cluster', data = X, markers = ['^','s','o'],
           fit_reg = False, scatter_kws = {'alpha':0.8}, legend = False)
plt.xlabel('评论')
plt.ylabel('销量')
plt.savefig('11.png', dpi=600)
plt.show()

sns.lmplot(x = '价格', y = '销量', hue = 'cluster', data = X, markers = ['^','s','o'],
           fit_reg = False, scatter_kws = {'alpha':0.8}, legend = False)
plt.xlabel('价格')
plt.ylabel('销量')
plt.savefig('12.png', dpi=600)
plt.show()



# 加载自定义词库
jieba.load_userdict(r'C:\Users\Lenovo\Desktop\all_words.txt')
# jieba.add_word('mylove')

#对评论进行分词, 并以空格隔开
df.word = df.title.apply(lambda x: ' '.join(jieba.cut(x)))
# df.content= df.content.apply(lambda x: ' '.join(SnowNLP(x).words) if len(x) > 0 else '')

# 读入停止词
with open(r'C:\Users\Lenovo\Desktop\stopwords.txt', encoding='UTF-8') as words:
    stop_words = [i.strip() for i in words.readlines()]

#分别去除cut_jieba和cut_snownlp中的停用词
df.words=df.word.apply(lambda x: ' '.join([w for w in (x.split(' ')) if w not in stop_words]))
# df.words = df.content.apply(lambda x: ' '.join([w for w in (x.split(' ')) if w not in stop_words]))

"""
评论分析
"""

texts = ';'.join(df.words.tolist())
# cut_text = " ".join(jieba.cut(texts))
# TF_IDF
keywords = jieba.analyse.extract_tags(texts, topK=200, withWeight=True, allowPOS=('a','e','n','nr','ns'))
text_cloud = dict(keywords)
pd.DataFrame(keywords).to_excel('TF_IDF关键词前200.xlsx')

# bg = plt.imread("abc.jpg")
# 生成
wc = WordCloud(# FFFAE3
    background_color="white",  # 设置背景为白色，默认为黑色
    width=1600,  # 设置图片的宽度
    height=1200,  # 设置图片的高度
    # mask=bg,
    max_words=2000,
    # stopwords={'春风十里不如你','亲亲','五十里','一百里'}
    margin=5,
    random_state = 2,
    max_font_size=500,  # 显示的最大的字体大小
    font_path="STSONG.TTF",
).generate_from_frequencies(text_cloud)
# 为图片设置字体

# 图片背景
# bg_color = ImageColorGenerator(bg)
# plt.imshow(wc.recolor(color_func=bg_color))
# plt.imshow(wc)
# 为云图去掉坐标轴
plt.axis("off")
plt.show()
wc.to_file("羽绒服.png")




# from pyecharts import Geo
# from collections import Counter
# #统计各地区出现次数, 并转换为元组的形式
# data = Counter(place).most_common()
# #生成地理坐标图
# geo =Geo("数据分析岗位各地区需求量", title_color="#fff", title_pos="center", width=1200, height=600, background_color='#404a59')
# attr, value =geo.cast(data)
# #添加数据点
# geo.add('', attr, value, visual_range=[0, 100],visual_text_color='#fff', symbol_size=5, is_visualmap=True, is_piecewise=True)
# geo.show_config()
# geo.render()


