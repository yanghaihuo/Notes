# -*- coding: utf-8 -*-
# import uniout  # 编码格式，解决中文输出乱码问题
import pandas as pd
from pyecharts import Pie,Line,Scatter
import os 
import numpy as np
import jieba
import jieba.analyse
from wordcloud import WordCloud,ImageColorGenerator
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import datetime

font = FontProperties(fname=r'c:\windows\fonts\simsun.ttc')# 指定本机的汉字字体位置

os.chdir(r'C:\Users\Lenovo\Desktop\工作细胞')

datas = pd.read_csv('bilibilib_gongzuoxibao.csv',index_col = 0,encoding = 'utf-8')


"""
描述性分析
"""


del datas['ctime']
del datas['cursor']
del datas['liked']




import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.pairplot(datas,kind="reg")
plt.show()



# 评分
scores = datas.score.groupby(datas['score']).count()
pie1 = Pie("评分", title_pos='center', width=900)
pie1.add(
    "评分",
    ['一星','二星','三星','四星','五星'],
    scores.values,
    radius=[40, 75],
#    center=[50, 50],
    is_random=True,
#    radius=[30, 75],
    is_legend_show=False,
    is_label_show=True,
)
pie1.render('评分.html')



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


num_date = datas.likes.groupby(datas['dates']).count()
# 点赞数时间分布
chart = Line("点赞数时间分布","666", title_color="#fff",title_pos="center", width=1200,height=600, background_color='#404a59')
chart.use_theme('dark')#使用主题
# bar.use_theme('light')
chart.add( '点赞数时间分布',num_date.index, num_date.values, is_fill=True, line_opacity=0.2,
          area_opacity=0.4, symbol=None)
# chart.print_echarts_options() # 该行只为了打印配置项，方便调试时使用
chart.render('点赞数时间分布.html')


# 好评字数分布
datalikes = datas.loc[datas.likes>5]
datalikes['num'] = datalikes.content.apply(lambda x:len(x))
chart = Scatter("likes")
chart.use_theme('dark')
chart.add('likes', np.log(datalikes.likes), datalikes.num, is_visualmap=True,
               xaxis_name = 'log(评论字数)',
          )
chart.render('好评字数(点赞数)分布.html')






# 点赞数每日内的时间分布
num_time = datas.likes.groupby(datas['time']).count()

# 时间分布
chart = Line("点赞数日内时间分布")
chart.use_theme('dark')
chart.add("点赞数", x_axis = num_time.index,y_axis = num_time.values,
          is_label_show=True,
          mark_point_symbol='diamond', mark_point_textcolor='#40ff27',
          line_width = 2
          )
chart.render('点赞数日内时间分布.html')


# 时间分布
chart = Line("点赞数时间分布")
chart.use_theme('dark')
chart.add( '点赞数时间分布',num_date.index, num_date.values, is_fill=True, line_opacity=0.2,
          area_opacity=0.4, symbol=None)

chart.render('点赞数时间分布.html')

# 评分时间分布
datascore = datas.score.groupby(datas.dates).mean()
chart = Line("评分时间分布")
chart.use_theme('dark')
chart.add('评分', datascore.index,
          datascore.values,
          line_width = 2
          )
chart.render('评分时间分布.html')


"""
评论分析
"""

texts = ';'.join(datas.content.tolist())
cut_text = " ".join(jieba.cut(texts))
# TF_IDF
keywords = jieba.analyse.extract_tags(cut_text, topK=500, withWeight=True, allowPOS=('a','e','n','nr','ns'))
text_cloud = dict(keywords)
pd.DataFrame(keywords).to_excel('TF_IDF关键词前500.xlsx')




bg = plt.imread("abc.jpg")
# 生成
wc = WordCloud(# FFFAE3
    background_color="white",  # 设置背景为白色，默认为黑色
    width=1600,  # 设置图片的宽度
    height=1200,  # 设置图片的高度
    mask=bg,
    max_words=2000,
    # stopwords={'春风十里不如你','亲亲','五十里','一百里'}
    margin=5,
    random_state = 2,
    max_font_size=500,  # 显示的最大的字体大小
    font_path="STSONG.TTF",
).generate_from_frequencies(text_cloud)
# 为图片设置字体

# 图片背景
bg_color = ImageColorGenerator(bg)
plt.imshow(wc.recolor(color_func=bg_color))
# plt.imshow(wc)
# 为云图去掉坐标轴
plt.axis("off")
plt.show()
wc.to_file("工作细胞词云.png")

# from pyecharts import WordCloud

# name = [
#     'Sam S Club', 'Macys', 'Amy Schumer', 'Jurassic World', 'Charter Communications',
#     'Chick Fil A', 'Planet Fitness', 'Pitch Perfect', 'Express', 'Home', 'Johnny Depp',
#     'Lena Dunham', 'Lewis Hamilton', 'KXAN', 'Mary Ellen Mark', 'Farrah Abraham',
#     'Rita Ora', 'Serena Williams', 'NCAA baseball tournament', 'Point Break']
# value = [
#     10000, 6181, 4386, 4055, 2467, 2244, 1898, 1484, 1112,
#    965, 847, 582, 555, 550, 462, 366, 360, 282, 273, 265]
# wordcloud = WordCloud(width=1300, height=620)
# wordcloud.add("", name, value, word_size_range=[20, 100])
# wordcloud.render("wordcloud.html")



# fig, ax = plt.subplots()
# ax.scatter(x=train['GrLivArea'], y=train['SalePrice'])
# plt.xlabel('GrLivArea', fontsize=13)
# plt.ylabel('SalePrice', fontsize=13)
# plt.show()



# bar = Bar("热评中点赞数示例图")
# bar.add("点赞数", user_list, likecount_list, is_stack=True, mark_line=["min", "max"], mark_point=["average"])
# bar.render()


# plt.figure(1,figsize=(10,12))
# plt.subplot(1,2,1)
# OL1_M[['Age']].boxplot(sym='m*',return_type = 'dict')
# plt.title('参赛男运动员箱体图')

# plt.subplot(1,2,2)
# OL1_F[['Age']].boxplot(sym='mo')
# plt.title('参赛女运动员箱体图')
# plt.show()

# plt.figure(1,figsize=(12,8))
# plt.subplot(2,2,1)
# OL2_summer_M['Age'].plot(kind='hist')
# plt.xlabel('夏季奥运会男运动员参赛年龄')
# plt.ylabel('频数')


# dist = sh.Dist.unique()
# plt.figure(1, figsize=(16, 30))
# with sns.axes_style("ticks"):
#     for i in range(17):
#         temp = sh[sh.Dist == dist[i]]
#         plt.subplot(6, 3, i + 1)
#         plt.title(dist[i])
#         sns.distplot(temp.Tprice)
#         plt.xlabel(' ')

# plt.show()

# temp = sh[sh.Dist == 'Xuhui']
# plt.figure(1,figsize=(6,6))
# plt.title('Xuhui')
# sns.distplot(temp.Tprice,kde=False,bins=20,rug=True)
# plt.xlabel(' ')
# plt.show()


# from scipy import stats, integrate
# plt.figure(1,figsize=(12,6))
# with sns.axes_style("ticks"):
#     plt.subplot(1,2,1)
#     sns.kdeplot(temp.Area,shade=True)
#     sns.rugplot(temp.Area)
#     plt.title('Xuhui --- Area Distribution')

# plt.subplot(1,2,2)
# plt.title('Xuhui - Area Distribution fits with gamma distribution')
# sns.distplot(temp.Area, kde=False, fit=stats.gamma)
# plt.show()

# bar = Bar("北京市各城区房屋平均单价","西城区最高,102276元/平米",width=1000,height=750,title_pos='center',jshost="https://cdn.kesci.com/nbextensions/echarts")
# bar.add('',d,uprice, mark_line=["average"], mark_point=["max", "min"],xaxis_interval=0,xaxis_rotate=65,is_more_utils=True)

# p1 = plt.figure(1,figsize=(6,6))
# coffee.Country.value_counts().head().plot(kind='bar',rot=0)
# coffee['Ownership Type'].value_counts().plot(kind='pie')
# plt.title('拥有星巴克门店最多的五个国家')
# plt.ylabel('Store Counts')
# plt.xlabel('Countries')
# plt.show()