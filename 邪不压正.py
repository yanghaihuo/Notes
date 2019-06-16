import matplotlib.pyplot as plt
from pyecharts import Pie,Line,Scatter,Bar,EffectScatter,Funnel,Overlap,Boxplot,Map,Geo,Style
import numpy as np
import jieba
import jieba.analyse
from wordcloud import WordCloud,ImageColorGenerator
import seaborn as sns
import pandas as pd
import scipy.stats as stats
import time
import os
import pymysql
from snownlp import SnowNLP
from sklearn.cluster import KMeans


# from matplotlib.font_manager import FontProperties
# myfont=FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf',size=16)
# sns.set(font=myfont.get_name())
# mpl.rcParams['font.sans-serif'] = ['SimHei']
# sns.set_style("whitegrid")
# sns.set_context('talk')
os.chdir(r'C:\Users\Lenovo\Desktop\xbsz')
# os.chdir(r'C:\Users\Lenovo\Desktop\cq')
# 可视化的中文处理
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')


#连接数据库
conn = pymysql.connect(host='localhost', user='root', password='Y1413010130H', port=3306, db='qgfx', charset='utf8')
sql = 'select * from xbsz'
#读取数据
df = pd.read_sql(sql, conn)

# df = pd.read_csv(r'C:\Users\Lenovo\Desktop\dzdp2luzhou_hotel.csv',encoding='gbk')

#设置行的最大显示为2条
# pd.set_option('display.max_rows', 2)
# print(df.head())
# print(df.shape)
# print(any(df.content.duplicated()))
# print(any(df.isnull()))
# print(df.nunique())
# print(df.isnull().sum())
# print(df.notnull().sum())
df.drop_duplicates(['userId','content'], inplace=True)
# df['createTime']=df.createTime.apply(lambda x: time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
df['createTime']=df.createTime.apply(lambda x:x[0:10] + '.' + x[11:13])
df['createTime']=df.createTime.astype('float')
df['createTime']=df.createTime.apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
df['createTime'] = pd.to_datetime(df['createTime'])
df['userLevel']=df.userLevel.astype('float')
df.drop('userIsLiked',axis=1,inplace=True)
# print(df.describe())
# print(df.describe(include=['object']))
df.info()
# print(df.shape)
# df.reset_index(drop=True,inplace=True)



# 1.   resample按时间聚合
# W星期,M月,Q季度,QS季度的开始第一天开始,A年,10A十年,10AS十年聚合日期第一天开始 H T S
#对去重后的数据按照天进行重新采样
# 首先要把索引变成时间
df.index = pd.DatetimeIndex(df['createTime'])
# 然后对其按照每天从新采样
df.D = df.content.resample('D').count()
df.D.plot(color='r',marker='D')
plt.title('每天评论数据')
plt.savefig('每天评论数据.png', dpi=600)
plt.show()
df.H = df.content.resample('H').count()
df.H.plot(color='g',marker='D',xticks=df.H.index)
plt.title('每小时评论数据')
plt.savefig('每小时评论数据.png', dpi=600)
plt.show()
# df.v = df.content.resample('T').count()
# df.v.plot(color='b',marker='D',xticks=df.v.index)
# plt.show()

# for x, y in zip(np.arange(len(df1.index)),df1.values):
#     plt.text(x, y, '%.0f' % y, ha='center', va='bottom')

# 2.   resample按时间聚合
# df.date = pd.to_datetime(df.date,format="%Y%m%d")
# df.set_index('date',drop=True)

# 常用日期处理方法
# dates = pd.to_datetime(pd.Series(['1989-8-18 13:14:55','1995-2-16']), format = '%Y-%m-%d %H:%M:%S')
# print('返回日期值：\n',dates.dt.date)
# print('返回季度：\n',dates.dt.quarter)
# print('返回几点钟：\n',dates.dt.hour)
# print('返回年中的天：\n',dates.dt.dayofyear)
# print('返回年中的周：\n',dates.dt.weekofyear)
# print('返回星期几的名称：\n',dates.dt.weekday_name)
# print('返回月份的天数：\n',dates.dt.days_in_month)
# datas['dates'] = pd.to_datetime(datas['date']).dt.date
# datas['time'] = pd.to_datetime(datas['date']).dt.hour
# datas['dates'] = pd.to_datetime(datas['date'],format='%Y年%m月')
# datas['dates'] = pd.to_datetime(datas['date'],format=''%Y-%m-%d %H:%M:%S'')


# month_message = db.groupby(['month'])
# month_com = month_message['box_office'].agg(['sum'])
# month_com.reset_index(inplace=True)
# month_com_last = month_com.sort_index()

# cd['area'].value_counts().plot(kind='bar')
# cd['area'].value_counts().plot(kind='hist')
# cd['area'].value_counts().plot(kind='pie')
# cd['area'].value_counts().plot(kind='kde')
# cd['area'].value_counts().plot(kind='barh')
# cd['area'].value_counts().plot(kind='box')
# cd['area'].value_counts().plot(kind='area')
# df4['a'].plot.hist(orientation='horizontal', cumulative=True)
# cd['area'].value_counts().plot(color='r',marker='D')
# baobao.plot(kind='scatter', x='price', y='new_estimate', color='g')
'''Series.plot（kind ='line'，ax = None，figsize = None，use_index = True，title = None，grid = None，legend = False，
  style = None，logx = False，logy = False，loglog = False，xticks = None，yticks = None，xlim = None，ylim = None，rot = None，
  fontsize = None，colormap = None，table = False，yerr = None，xerr = None，label = None，secondary_y = False，** kwds ）'''


df.reset_index(drop=True,inplace=True)

# df['userLevel'].plot(kind='hist',orientation='horizontal', cumulative=True)
df['userLevel'].plot(kind='hist',normed = True,cumulative=True,color='steelblue')
# 添加水平参考线
plt.axhline(y = 0.1, color = 'yellow',linestyle = '--', linewidth = 2)
plt.axhline(y = 0.3, color = 'blue', linestyle = '--', linewidth = 2)
plt.axhline(y = 0.5, color = 'red', linestyle = '--', linewidth = 2)
plt.axhline(y = 0.7, color = 'green', linestyle = '--', linewidth = 2)
plt.title('用户等级')
plt.savefig('用户等级累计直方图.png', dpi=600)
plt.show()

# 绘图：乘客年龄的累计频率直方图
# plt.hist(titanic.Age, # 绘图数据
#         bins = np.arange(titanic.Age.min(),titanic.Age.max(),5), # 指定直方图的组距
#         normed = True, # 设置为频率直方图
#         cumulative = True, # 积累直方图
#         color = 'steelblue', # 指定填充色
#         edgecolor = 'k', # 指定直方图的边界色
#         label = '直方图' )# 为直方图呈现标签

# df.groupby('A').agg({'B': ['min', 'max'], 'C': 'sum'})
df1 = df["upCount"].groupby(df['content']).sum().sort_values(ascending=False)
df2 = df["downCount"].groupby(df['content']).sum().sort_values(ascending=False)
df3 = df["replyCount"].groupby(df['content']).sum().sort_values(ascending=False)
print(df1)
print(df2)
print(df3)
# sns.barplot(df1.index,df1.values)
# plt.xticks(np.arange(len(df1.index)),df1.index, rotation = 90)
# for x, y in zip(np.arange(len(df1.index)),df1.values):
#     plt.text(x, y, '%.0f' % y, ha='center', va='bottom')
# plt.title('泸州各地区酒店均价前二十分布')
# plt.savefig('泸州各地区酒店均价前二十分布.png', dpi=600)
# plt.show()
# 运用正则表达式，将评论中的数字和英文去除
df.content = df.content.str.replace('[0-9a-zA-Z]', '')
df.content=df.content.str.replace("1f\d.+",'')

# signatures = ''
emotions = []
for signatures in df.content:
    if (signatures != None):
        signatures = signatures.strip()
        if (len(signatures) > 0):
            nlp = SnowNLP(signatures)
            # print(nlp)
            emotions.append(nlp.sentiments)
            # signatures += ' '.join(jieba.analyse.extract_tags(signature, 5))

# df.content= df.content.apply(lambda x: SnowNLP(x).sentiments if len(x) > 0 else '')
# print(df.content)
# Signature Emotional Judgment
count_good = len(list(filter(lambda x:x>0.66,emotions)))
# print(list(filter(lambda x:x>0.66,emotions)))
count_normal = len(list(filter(lambda x:x>=0.33 and x<=0.66,emotions)))
count_bad = len(list(filter(lambda x:x<0.33,emotions)))
labels = [u'负面消极',u'中性',u'正面积极']
values = (count_bad,count_normal,count_good)
plt.xlabel(u'情感判断')
plt.ylabel(u'频数')
plt.xticks(range(3),labels)
plt.legend(loc='upper right',)
plt.bar(range(3), values, color = 'rgb')
plt.title('邪不压正评论的情感分析')
plt.savefig('邪不压正评论的情感分析.png', dpi=600)
plt.show()

# 加载自定义词库
jieba.load_userdict(r'C:\Users\Lenovo\Desktop\all_words.txt')
# jieba.add_word('mylove')

#对评论进行分词, 并以空格隔开
df.word = df.content.apply(lambda x: ' '.join(jieba.cut(x)))
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
wc.to_file("邪不压正.png")



'''
a:形容词；
b:连词；
d:副词；
e:叹词；
f:方位词；
i:成语；
m:数词；
n:名词；
nr：人名；
ns：地名；
nt:机构团体；
p:介词；
r:代词；
t:时间；
u:助词；
v:动词；
vn:名动词；
w:标点符号；
un：未知词语；
'''

