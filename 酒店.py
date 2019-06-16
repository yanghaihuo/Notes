import matplotlib.pyplot as plt
from pyecharts import Pie,Line,Scatter,Bar,EffectScatter,Funnel,Overlap,Boxplot,Map,Geo,Style
import numpy as np
import jieba
import jieba.analyse
from wordcloud import WordCloud,ImageColorGenerator
import seaborn as sns
import pandas as pd
import scipy.stats as stats
import os
import pymysql
from sklearn.cluster import KMeans


# from matplotlib.font_manager import FontProperties
# myfont=FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf',size=16)
# sns.set(font=myfont.get_name())
# mpl.rcParams['font.sans-serif'] = ['SimHei']
# sns.set_style("whitegrid")
# sns.set_context('talk')
os.chdir(r'C:\Users\Lenovo\Desktop\lz')
# os.chdir(r'C:\Users\Lenovo\Desktop\cq')
# 可视化的中文处理
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')



# #连接数据库
# conn = pymysql.connect(host='localhost', user='root', password='Y1413010130H', port=3306, db='cd_hotel', charset='utf8')
# sql = 'select * from cd'
# #读取数据
# df = pd.read_sql(sql, conn)


df = pd.read_csv(r'C:\Users\Lenovo\Desktop\dzdp2luzhou_hotel.csv',encoding='gbk')



#设置行的最大显示为2条
# pd.set_option('display.max_rows', 2)
# print(df.head())
# print(df.shape)
# print(any(df.hotel.duplicated()))
# print(any(df.isnull()))
# print(df.nunique())
# print(df.isnull().sum())
# print(df.notnull().sum())
df.drop_duplicates('hotel', inplace=True)
df['comment']=df.comment.str[1:-1].astype('float')
# print(df.describe())
# print(df.describe(include=['object']))
df.info()
# print(df.shape)
df.reset_index(drop=True,inplace=True)

df['location']=df['Small_area'].replace(['钟鼓楼','迎晖路','水井沟','大山坪','百子图','酒城乐园','酒城大道','东门口','巴士花园','肖巷子','佳乐广场','龙透关','市中心/水井沟'],'江阳区')
df['location']=df['location'].replace(['回龙湾','小市','龙马大道','科维商城','华润万象汇','春雨路','红星村','龙南路/回龙湾','老窖广场/巨洋大剧院','杜家街','西南商贸城'],'龙马潭区')
df['location']=df['location'].replace('泸县其他','泸县')
df['location']=df['location'].replace(['合江县其他','九支镇','榕山镇'],'合江县')
df['location']=df['location'].replace(['叙永县其他','叙永镇'],'叙永县')
df['location']=df['location'].replace(['古蔺县其他','二郎镇'],'古蔺县')


# food = {'<=100': '1', '100-500': '2', '500-1000': '3', '1000-3000': '4', '>3000': '5'}
# df['工作日消费指数'] = df['工作日消费金额'].map(food)

# cd=cd.loc[cd.price!='价格待定',:]
# df=df.loc[df['classification'].isin(['火锅']),:]
# y=lambda x:float(x[:-1]) * 10000 if x.find('万') else float(x)
# df["Small_area"]=df.Small_area.apply(lambda x:x[1:-1])
# df.reset_index(drop=True,inplace=True)# inplace 和 赋值不能同时用
# df.reset_index(inplace=True)
# df.index=range(0,df.shape[0])


# import warnings
# warnings.filterwarnings("ignore")
# sns.pairplot(df,palette="Set2",kind="reg")
# plt.show()


# # 散点图
# sns.relplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', style='Species', data=df_Iris )
# plt.title('SepalLengthCm and SepalWidthCm data by Species')
# # 散点图加直方图
# sns.jointplot(x='SepalLengthCm', y='SepalWidthCm', data=df_Iris)
# # 质量分布图
# sns.distplot(df.price, bins=10, hist=True, kde=True)

sns.distplot(df.price, hist=True, kde=True)
plt.title('泸州酒店价格区间分布')
plt.savefig('泸州酒店价格区间分布.png', dpi=600)
plt.show()

sns.distplot(df.comment, hist=True, kde=True)
plt.title('泸州酒店评论数分布')
plt.savefig('泸州酒店评论数分布.png', dpi=600)
plt.show()

star = df['star_level'].value_counts()
plt.pie(star.values, labels=star.index, autopct='%.2f %%')
plt.title('酒店星级分布')
# df['star_level'].value_counts().plot(kind='pie')
plt.savefig('酒店星级分布.png', dpi=600)
plt.show()


# # 箱线图
# sns.boxplot(x='price', y='comment',hue='Species', data=Iris)
# # 琴图(箱线图与核密度图)
# sns.violinplot(x='Attribute', y='Data', hue='Species', data=Iris )
# # 多变量图
# sns.pairplot(df_Iris.drop('Id', axis=1), hue='Species',kind="reg")
# # 直方图
# sns.barplot(data["star"],y=data["comment"],order = ['五星商户','准五星商户','四星商户','准四星商户','三星商户'])
# 柱状图
# sns.countplot(x="salary",hue="department",data=df)
# 折线图
# sns.pointplot(x='time_spend_company',y='left',data=df)
# 饼图
# lbs = df['salary'].value_counts().index
# explodes=[0.1 if i=="high" else 0 for i in lbs]
# plt.pie(df['salary'].value_counts(normalize=True),explode=explodes,
#         labels=lbs,autopct="%1.1f%%",colors=sns.color_palette("Reds"))



df1 = df["price"].groupby(df['location']).mean().sort_values(ascending=False)
sns.barplot(df1.index,df1.values)
plt.xticks(np.arange(len(df1.index)),df1.index, rotation = 90)
for x, y in zip(np.arange(len(df1.index)),df1.values):
    plt.text(x, y, '%.0f' % y, ha='center', va='bottom')
plt.title('泸州各区县酒店均价分布')
plt.savefig('泸州各区县酒店均价分布.png', dpi=600)
plt.show()


df1 = df["price"].groupby(df['location']).count().sort_values(ascending=False)
sns.barplot(df1.index,df1.values)
plt.xticks(np.arange(len(df1.index)),df1.index, rotation = 90)
for x, y in zip(np.arange(len(df1.index)),df1.values):
    plt.text(x, y, '%.0f' % y, ha='center', va='bottom')
plt.title('泸州各区县酒店数量分布')
plt.savefig('泸州各区县酒店数量分布.png', dpi=600)
plt.show()


df1 = df["price"].groupby(df['Small_area']).mean().sort_values(ascending=False)[:20]
sns.barplot(df1.index,df1.values)
plt.xticks(np.arange(len(df1.index)),df1.index, rotation = 90)
for x, y in zip(np.arange(len(df1.index)),df1.values):
    plt.text(x, y, '%.0f' % y, ha='center', va='bottom')
plt.title('泸州各地区酒店均价前二十分布')
plt.savefig('泸州各地区酒店均价前二十分布.png', dpi=600)
plt.show()


df2 = df["hotel"].groupby(df['Small_area']).count().sort_values(ascending=False)[:20]
sns.barplot(df2.index,df2.values)
plt.xticks(np.arange(len(df2.index)),df2.index, rotation = 90)
for x, y in zip(np.arange(len(df2.index)),df2.values):
    plt.text(x, y, '%.0f' % y, ha='center', va='bottom')
plt.title('泸州各地区酒店数量前二十分布')
plt.savefig('泸州各地区酒店数量前二十分布.png', dpi=600)
plt.show()


df2 = df["price"].groupby(df['style']).mean()
sns.barplot(df2.index,df2.values)
plt.xticks(np.arange(len(df2.index)),df2.index, rotation = 90)
for x, y in zip(np.arange(len(df2.index)),df2.values):
    plt.text(x, y, '%.0f' % y, ha='center', va='bottom')
plt.title('泸州各风格酒店均价分布')
plt.savefig('泸州各风格酒店均价分布.png', dpi=600)
plt.show()


df2 = df["hotel"].groupby(df['style']).count()
sns.barplot(df2.index,df2.values)
plt.xticks(np.arange(len(df2.index)),df2.index, rotation = 90)
for x, y in zip(np.arange(len(df2.index)),df2.values):
    plt.text(x, y, '%.0f' % y, ha='center', va='bottom')
plt.title('泸州各风格酒店数量分布')
plt.savefig('泸州各风格酒店数量分布.png', dpi=600)
plt.show()


sns.lmplot(x='comment', y='price', hue='star_level',data=df, fit_reg=False, scatter=True)
plt.title('价格-评论数关系')
plt.savefig('价格-评论数关系.png', dpi=600)
plt.show()


sns.boxplot(df["star_level"],df["price"],palette="Set3")
sns.stripplot(x="star_level", y="price", data=df, alpha=".25")
plt.title('价格-星级关系')
plt.savefig('价格-星级关系.png', dpi=600)
plt.show()


sns.violinplot(x='star_level', y='price',data=df, palette="Set2",fit_reg=False)
plt.title('价格-星级关系1')
plt.savefig('价格-星级关系1.png', dpi=600)
plt.show()


df3 = df["comment"].groupby(df['star_level']).sum()
sns.barplot(df3.index,df3.values)
for x, y in zip(np.arange(len(df3.index)),df3.values):
    plt.text(x, y, '%.0f' % y, ha='center', va='bottom')
plt.title('星级-评论数关系')
plt.savefig('星级-评论数关系.png', dpi=600)
plt.show()


df4 = df[['star_level','price', 'comment']]
sns.pairplot(df4,hue='star_level',palette="Set2",kind="reg")
plt.title('星级-价格-评论数关系')
plt.savefig('星级-价格-评论数关系.png', dpi=600)
plt.show()



from collections import Counter
estimator = KMeans(n_clusters=3)#构造聚类器
data_1 = df[['price', 'comment']]
estimator.fit(data_1)#聚类
label_pred = estimator.labels_ #获取聚类标签
centroids = estimator.cluster_centers_ #获取聚类中心
inertia = estimator.inertia_ # 获取聚类准则的总和
print(label_pred)
print(Counter(label_pred))
print(centroids)
print(inertia)


# #绘制k-means结果
x0 = data_1[label_pred == 0]
x1 = data_1[label_pred == 1]
x2 = data_1[label_pred == 2]
plt.figure(figsize=(8,8))
plt.scatter(x0['price'], x0['comment'], c = "red", marker='o', label='低评论')
plt.scatter(x1['price'], x1['comment'], c = "green", marker='*', label='中间评论')
plt.scatter(x2['price'], x2['comment'], c = "blue", marker='+', label='高评论')
plt.xlabel('price')
plt.ylabel('comment')
plt.legend()
plt.title('价格-评论数关系')
plt.savefig('价格-评论数关系.png', dpi=600)
plt.show()

# plt.figure(figsize=(8,8))
# plt.scatter(x0['taste'], x0['services'], c = "red", marker='o', label='一般推荐')
# plt.scatter(x1['taste'], x1['services'], c = "green", marker='*', label='不推荐')
# plt.scatter(x2['taste'], x2['services'], c = "blue", marker='+', label='非常推荐')
# plt.xlabel('taste')
# plt.ylabel('services')
# plt.legend()
# plt.title('口味与服务')
# plt.show()


# cd.price = cd.price.str.extract('(.*?)元/平')
df["stay"] = df["stay_time"].str.extract('请在(.*?)以后入住')
df['leave'] = df["stay_time"].str.extract('次日(.*?)前退房')
df7 = df["Opening_decoration"].str.extract('(\d+)年开业')
df8 = df["Opening_decoration"].str.extract('(\d+)年装修')
print(df.stay)


df.stay_time.value_counts().plot(kind='barh',rot=0)
plt.title('酒店入住和离开时间')
plt.savefig('酒店入住和离开时间.png', dpi=600)
plt.show()


df.stay.value_counts().plot(kind='barh',rot=0)
plt.title('酒店入住时间')
plt.savefig('酒店入住时间.png', dpi=600)
plt.show()


df.leave.value_counts().plot(kind='barh',rot=0)
plt.title('离开酒店时间')
plt.savefig('离开酒店时间.png', dpi=600)
plt.show()




# # 导入绘图模块
# import matplotlib.pyplot as plt
# # 构建数据
# price = [39.5,39.9,45.4,38.9,33.34]
# # 绘图
# plt.barh(range(5), price, align = 'center',color='steelblue', alpha = 0.8)
# # 添加轴标签
# plt.xlabel('价格')
# # 添加标题
# plt.title('不同平台书的最低价比较')
# # 添加刻度标签
# plt.yticks(range(5),['亚马逊','当当网','中国图书网','京东','天猫'])
# # 设置Y轴的刻度范围
# plt.xlim([32,47])
#
# # 为每个条形图添加数值标签
# for x,y in enumerate(price):
#     plt.text(y+0.1,x,'%s' %y,va='center')
# # 显示图形
# plt.show()


"""
文本分析
"""

texts = ';'.join(df.Hotel_Amenities.dropna().tolist())
# texts = ';'.join(df.Hotel_Amenities.tolist())
cut_text = " ".join(jieba.cut(texts))
# TF_IDF
keywords = jieba.analyse.extract_tags(cut_text, topK=100, withWeight=True, allowPOS=('a','e','n','nr','ns'))
text_cloud = dict(keywords)
pd.DataFrame(keywords).to_excel('TF_IDF关键词1前100.xlsx')
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
wc.to_file("酒店设施.png")




# # 导入第三方包
# import jieba
#
# # 加载自定义词库
# jieba.load_userdict(r'C:\Users\Lenovo\Desktop\all_words.txt')
#
# # 读入停止词
# with open(r'C:\Users\Lenovo\Desktop\mystopwords.txt', encoding='UTF-8') as words:
#     stop_words = [i.strip() for i in words.readlines()]
#
# # 构造切词的自定义函数，并在切词过程中删除停止词
# def cut_word(sentence):
#     words = [i for i in jieba.lcut(sentence) if i not in stop_words]
#     # 切完的词用空格隔开
#     result = ' '.join(words)
#     return(result)
# # 对评论内容进行批量切词
# words = evaluation.Content.apply(cut_word)
# # 前5行内容的切词效果
# print(words[:5])



texts = ';'.join(df.brief_introduction.dropna().tolist())
# texts = ';'.join(df.brief_introduction.tolist())
cut_text = " ".join(jieba.cut(texts))
# TF_IDF
keywords = jieba.analyse.extract_tags(cut_text, topK=100, withWeight=True, allowPOS=('a','e','n','nr','ns'))
text_cloud = dict(keywords)
pd.DataFrame(keywords).to_excel('TF_IDF关键词2前100.xlsx')
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
wc.to_file("酒店简介.png")











# # 星级
# # scores = df[['star_level','comment']].groupby(df['star_level']).count()
# scores = df['star_level'].groupby(df['star_level']).count()
# pie1 = Pie("星级", '各星级火锅店的数量占比', title_pos='center', width=800,height=800)
# pie1.add(
#     "",
#     scores.index,
#     scores.values,
#     radius=[40, 75],
# #    center=[50, 50],
#     is_random=True,
# #    radius=[30, 75],
#     is_legend_show=False,
#     is_label_show=False,
# )
# pie1.render('星级.html')
#
#
# num_date = df.comment.groupby(df['star_level']).sum()
# chart = Bar("各星级下的评论数分布","", title_color="#fff",title_pos="center", width=1200,height=600, background_color='#404a59')
# chart.use_theme('dark')#使用主题
# # chart.use_theme('light')
# chart.add( '',num_date.index, num_date.values,mark_point=["min", "max","average"],is_legend_show=False,is_label_show=False)
# # chart.print_echarts_options() # 该行只为了打印配置项，方便调试时使用
# chart.render('各星级下的评论数分布.html')
#
#
#
# num0 = df.consumption.groupby(df['Small_area']).mean().sort_values(ascending=False)
# bar = Bar('各地区火锅均价')
# # bar.use_theme('dark')#使用主题
# # bar.use_theme('light')
# bar.add( '直方图',num0.index, num0.values,mark_point=["min", "max","average"],mark_line=["min", "max","average"])
# # chart.print_echarts_opti.ons() # 该行只为了打印配置项，方便调试时使用
# line = Line()
# line.add( '',num0.index, num0.values, is_fill=True, line_opacity=0.2,mark_point=["min", "max","average"],mark_line=["min", "max"],
#           area_opacity=0.4, symbol=None)
# # line.add( '折线图',num0.index, num0.values,is_smooth=True,mark_point=["min", "max","average"])# is_label_show=True
# overlap = Overlap(width=1500, height=800)
# overlap.add(bar)
# overlap.add(line, yaxis_index=1, is_add_yaxis=True)
# overlap.render('各地区火锅均价折线直方图.html')
#
#
#
# num = df.consumption.groupby(df['Small_area']).mean().sort_values(ascending=False)[:20]
# # 点赞数时间分布
# chart = Bar("均价前二十各地区的分布","", title_color="#fff",title_pos="center", width=1500,height=600, background_color='#404a59')
# chart.use_theme('dark')#使用主题
# # chart.use_theme('light')
# chart.add( '',num.index, num.values, is_fill=True, line_opacity=0.2,mark_point=["min", "max","average"],mark_line=["min", "max"],
#           area_opacity=0.4, symbol=None)
# # chart.print_echarts_options() # 该行只为了打印配置项，方便调试时使用
# chart.render('均价前二十各地区的分布图.html')
#
#
# num1= df.consumption.groupby(df['Small_area']).mean().sort_values()[:20]
# # 点赞数时间分布
# chart = Bar("均价后二十各地区的分布","", title_color="#fff",title_pos="center", width=1500,height=600, background_color='#404a59')
# chart.use_theme('dark')#使用主题
# # chart.use_theme('light')
# chart.add( '',num1.index, num1.values, is_fill=True, line_opacity=0.2,
#           area_opacity=0.4, symbol=None)
# # chart.print_echarts_options() # 该行只为了打印配置项，方便调试时使用
# chart.render('均价后二十各地区的分布图.html')
#
#
# num2 = df.comment.groupby(df['consumption']).sum()
# chart = Line("各价格下的评论数分布","", title_color="#fff",title_pos="center", width=1200,height=600, background_color='#404a59')
# chart.use_theme('dark')#使用主题
# # bar.use_theme('light')
# chart.add( '',num2.index, num2.values, is_fill=True, line_opacity=0.2,
#           area_opacity=0.4, symbol=None,mark_point=["min", "max","average"],xaxis_name = '价格',yaxis_name = '评论数')
# # chart.print_echarts_options() # 该行只为了打印配置项，方便调试时使用
# chart.render('各价格下的评论数分布.html')
#
#
# chart = Scatter("口味·环境散点图" ,title_color="",title_pos="center")
# chart.use_theme('light')
# chart.add('', df.taste, df.environment, is_visualmap=True,
#                xaxis_name = '环境',yaxis_name = '口味'
#           )
# chart.render('口味·环境散点图分布.html')
#
#
# num3=df.region_name.value_counts()
# funnel = Funnel('地区火锅数量漏斗图', title_pos='center')
# funnel.add("", num3.index, num3.values, is_label_show=True, label_pos="inside", label_text_color="#fff", legend_pos="left", legend_orient="vertical")
# funnel.render('地区火锅数量漏斗图分布.html')
# # boxplot=Boxplot("1、2班考试成绩比较","箱线图",title_pos="center",width=1200,height=800)
#
#
#
# df.dropna(inplace=True)
# num4 = df.consumption.values
# num5 = df.comment.values
# boxplot = Boxplot('火锅价格·评论箱体图')
# x_axis = ['价格','评论']
# y_axis = [num4,num5]
# yaxis = boxplot.prepare_df(y_axis)
# boxplot.add("",x_axis,yaxis)
# # boxplot.add("",num4.index,num4.values,is_label_show=True, label_pos="inside", label_text_color="#fff", legend_pos="left", legend_orient="vertical")
# boxplot.render('火锅价格·评论箱体图分布.html')
#
#
#
# num4 = df.consumption.values
# boxplot = Boxplot('火锅价格箱体图')
# x_axis = ['价格']
# y_axis = [num4]
# yaxis = boxplot.prepare_df(y_axis)
# boxplot.add("",x_axis,yaxis)
# # boxplot.add("",num4.index,num4.values,is_label_show=True, label_pos="inside", label_text_color="#fff", legend_pos="left", legend_orient="vertical")
# boxplot.render('火锅价格箱体图分布.html')
#
#
# num6 = df.consumption.groupby(df['shop']).mean().sort_values(ascending=False)
# chart = Bar("所有火锅店的价格分布","", title_color="#fff",title_pos="center", width=1500,height=600, background_color='#404a59')
# chart.use_theme('dark')#使用主题
# # chart.use_theme('light')
# chart.add( '',num6.index, num6.values,mark_point=["min", "max","average"],mark_line=["min", "max"])
# # chart.print_echarts_options() # 该行只为了打印配置项，方便调试时使用
# chart.render('所有火锅店的价格分布.html')
#
#
# # df = df.corr()
# # sns.heatmap(df)
# # plt.show()
#
#
# # 绘制直方图
# df.consumption.plot(kind = 'hist', bins = 30, normed = True)
# # 绘制核密度图
# df.consumption.plot(kind = 'kde')
# plt.savefig('质量分布图.png', dpi=600)
# # 图形展现
# plt.show()
#
#
# sns.distplot(a = df.consumption, bins = 10, fit = stats.norm, norm_hist = True,
#              hist_kws = {'color':'steelblue', 'edgecolor':'black'},
#              kde_kws = {'color':'black', 'linestyle':'--', 'label':'核密度曲线'},
#              fit_kws = {'color':'red', 'linestyle':':', 'label':'正态密度曲线'})
# plt.savefig('质量分布图1.png', dpi=600)
# plt.show()
#
#
#
# df1=df.loc[df.consumption >= 200,:]
# num7 = df1.consumption.groupby(df1['shop']).mean().sort_values(ascending=False)
# chart = Bar("价格大于200的火锅店","", title_color="#fff",title_pos="center", width=1500,height=600, background_color='#404a59')
# chart.use_theme('dark')#使用主题
# # chart.use_theme('light')
# chart.add( '',num7.index, num7.values,mark_point=["min", "max","average"],mark_line=["min", "max"])
# # chart.print_echarts_options() # 该行只为了打印配置项，方便调试时使用
# chart.render('价格大于200的火锅店.html')
#
#
#
# # try:
# # districts = ['运河区', '新华区', '泊头市', '任丘市', '黄骅市', '河间市', '沧县', '青县', '东光县', '海兴县', '盐山县', '肃宁县', '南皮县', '吴桥县', '献县', '孟村回族自治县']
# # areas = [109.92, 109.47, 1006.5, 1023.0, 1544.7, 1333.0, 1104.0, 968.0, 730.0, 915.1, 796.0, 525.0, 794.0, 600.0, 1191.0, 387.0]
# num8 = df.shop.groupby(df['region_name']).count()
# map_1 = Map("各地区火锅数量", width=1600, height=900)
# map_1.add("", num8.index, num8.values, maptype='重庆', is_visualmap=True, visual_range=[min(num0.values), max(num0.values)],
#         visual_text_color='#000', is_map_symbol_show=False, is_label_show=True)
# map_1.render('各地区火锅数量地图.html')
#
#
# style = Style(title_color="#fff", title_pos="center",
#               width=1200, height=600, background_color="#404a59")
# map_2 = Map('各地区火锅数量', **style.init_style)
# map_2.add("", num8.index, num8.values, visual_range=[0, 100],
#         visual_text_color="#fff", type='heatmap',
#         is_visualmap=True,is_piecewise = True, maptype='重庆'
#         )
#         # geo_cities_coords=geo_cities_coords)
# map_2.render('各地区火锅数量热力图.html')


# except ValueError as e:
#     pass
    # e = str(e)
    # e = e.split("No coordinate is specified for ")[1]#获取不支持的城市名
    # for i in range(0,len(map_1)):
    #     if e in list(map_1[i]):
    #         del map_1[i]
    #         break



# # 点赞数每日内的时间分布
# num_time = dfs.likes.groupby(dfs['time']).count()
#
# # 时间分布
# chart = Line("点赞数日内时间分布")
# chart.use_theme('dark')
# chart.add("点赞数", x_axis = num_time.index,y_axis = num_time.values,
#           is_label_show=True,
#           mark_point_symbol='diamond', mark_point_textcolor='#40ff27',
#           line_width = 2
#           )
# chart.render('点赞数日内时间分布.html')
#
#
# # 时间分布
# chart = Line("点赞数时间分布")
# chart.use_theme('dark')
# chart.add( '点赞数时间分布',num_date.index, num_date.values, is_fill=True, line_opacity=0.2,
#           area_opacity=0.4, symbol=None)
#
# chart.render('点赞数时间分布.html')
#
# # 评分时间分布
# dfscore = dfs.score.groupby(dfs.dates).mean()
# chart = Line("评分时间分布")
# chart.use_theme('dark')
# chart.add('评分', dfscore.index,
#           dfscore.values,
#           line_width = 2
#           )
# chart.render('评分时间分布.html')
#
#
# """
# 评论分析
# """
#
# texts = ';'.join(dfs.content.tolist())
# cut_text = " ".join(jieba.cut(texts))
# # TF_IDF
# keywords = jieba.analyse.extract_tags(cut_text, topK=500, withWeight=True, allowPOS=('a','e','n','nr','ns'))
# text_cloud = dict(keywords)
# pd.DataFrame(keywords).to_excel('TF_IDF关键词前500.xlsx')
#
#
#
#
# bg = plt.imread("abc.jpg")
# # 生成
# wc = WordCloud(# FFFAE3
#     background_color="white",  # 设置背景为白色，默认为黑色
#     width=1600,  # 设置图片的宽度
#     height=1200,  # 设置图片的高度
#     mask=bg,
#     max_words=2000,
#     # stopwords={'春风十里不如你','亲亲','五十里','一百里'}
#     margin=5,
#     random_state = 2,
#     max_font_size=500,  # 显示的最大的字体大小
#     font_path="STSONG.TTF",
# ).generate_from_frequencies(text_cloud)
# # 为图片设置字体
#
# # 图片背景
# bg_color = ImageColorGenerator(bg)
# plt.imshow(wc.recolor(color_func=bg_color))
# # plt.imshow(wc)
# # 为云图去掉坐标轴
# plt.axis("off")
# plt.show()
# wc.to_file("工作细胞词云.png")

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
# sns.set_style("whitegrid")
# fig,axes=plt.subplots(1,3) #创建一个一行三列的画布
# sns.distplot(cd.price,fit = stats.norm,norm_hist = True,ax=axes[0]) #左图
# sns.distplot(cd.price,hist=False,ax=axes[1]) #中图
# sns.distplot(cd.price,kde=False,ax=axes[2]) #右图
# plt.show()
#
# sns.distplot(cd.price,color='steelblue')
# plt.show()

# fig,axes=plt.subplots(1,2)
# sns.distplot(cd.price,norm_hist=True,kde=False,ax=axes[0]) #左图 密度
# sns.distplot(cd.price,kde=False,ax=axes[1]) #右图 计数
# plt.show()





# df = baobao.corr()
# sns.heatmap(df)
# plt.show()

# print(baobao['new_estimate'].sort_values())


# sns.lmplot(x='price', y='new_estimate',df=baobao,fit_reg=True, scatter=True ) #散点图
# plt.show()
# sns.lmplot(x='price', y='new_estimate',hue='shop',df=baobao,fit_reg=False, scatter=True)
# plt.show()

# train = train[-((train.SalePrice < 200000) &  (train.GrLivArea > 4000))]

# sns.jointplot(x='Area',y='Tprice',df=sh) #散点图
# sns.jointplot(x='Area',y='Tprice',df=sh，kind='hex')
# plt.show()


# lm = sns.lmplot(x = 'Age', y = 'Fare', df = titanic, hue = 'Sex', fit_reg=True)
# lm.set(title = 'Fare x Age')
# axes = lm.axes
# axes[0,0].set_ylim(-5,)
# axes[0,0].set_xlim(-5,85)


# from os import path

# d=path.dirname(__file__)


# df.drop(df[df['职位名称'].str.contains('实习')].index, inplace=True)
# train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)






# gender_map = {0: 'unknown', 1: 'male', 2: 'female'}
# df['gender'] = df['gender'].apply(lambda x: gender_map[x]) #映射
# df.sample(5) #随机选5行

# newdfset = dfset[-dfset['城区'].isin(['燕郊'])]



# plt.subplot2grid((2,3),(1,0), colspan=2)  # colspan = 2 表示横向跨度是 2
# plt.figure(1, figsize=(40, 60))
# plt.subplot(2,2,1)
# cd['area'].value_counts().plot(kind='bar')
# plt.subplot(2,2,2)
# cd['area'].value_counts().plot(kind='hist')
# plt.subplot(2,2,3)
# cd['area'].value_counts().plot(kind='pie')
# plt.subplot(2,2,4)
# cd['area'].value_counts().plot(kind='kde')
# plt.show()
# # plt.subplot(3,2,5)
# cd['area'].value_counts().plot(kind='barh')
# plt.show()
# cd['area'].value_counts().plot(kind='box')
# plt.show()
# cd['area'].value_counts().plot(kind='area')
# plt.show()
# cd['area'].value_counts().plot(color='r',marker='D')
# plt.show()
# baobao.plot(kind='scatter', x='price', y='new_estimate', color='g')
# plt.show()
# '''Series.plot（kind ='line'，ax = None，figsize = None，use_index = True，title = None，grid = None，legend = False，
#   style = None，logx = False，logy = False，loglog = False，xticks = None，yticks = None，xlim = None，ylim = None，rot = None，
#   fontsize = None，colormap = None，table = False，yerr = None，xerr = None，label = None，secondary_y = False，** kwds ）'''
#
#
#
# import scipy.stats as stats
# # 绘制直方图
# sns.distplot(a = cd['area'].value_counts(), bins = 10, fit = stats.norm, norm_hist = True,
#              hist_kws = {'color':'steelblue', 'edgecolor':'black'},
#              kde_kws = {'color':'black', 'linestyle':'--', 'label':'核密度曲线'},
#              fit_kws = {'color':'red', 'linestyle':':', 'label':'正态密度曲线'})
# # 显示图例
# plt.legend()
# # 显示图形
# plt.show()
#
# # 导入绘图模块
# import matplotlib.pyplot as plt
# # 设置绘图风格
# plt.style.use('ggplot')
# # 绘制直方图
# sunspots.counts.plot(kind = 'hist', bins = 30, normed = True)
# # 绘制核密度图
# sunspots.counts.plot(kind = 'kde')
# # 图形展现
# plt.show()