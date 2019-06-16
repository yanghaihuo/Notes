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

os.chdir(r'C:\Users\Lenovo\Desktop\cd')
# os.chdir(r'C:\Users\Lenovo\Desktop\cq')
# 可视化的中文处理
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')
data = pd.read_csv(r'C:\Users\Lenovo\Desktop\dzdpcdhuoguo.csv',encoding='gbk')
# data = pd.read_csv(r'C:\Users\Lenovo\Desktop\dzdpcqhuoguo.csv',encoding='gbk')
print(data.shape)
# print(any(data.location.isnull()))
# data.drop_duplicates('shop', inplace=True)
# print(data.shape)
data['consumption']=data.consumption.str[1:].astype('float')
# print(data.nunique())
# print(data.isnull().sum())
# print(data.notnull().sum())
# print(data.head())
# cd=cd.loc[cd.price!='价格待定',:]
data=data.loc[data['classification'].isin(['火锅']),:]
print(data.shape)
# y=lambda x:float(x[:-1]) * 10000 if x.find('万') else float(x)
data["Small_area"]=data.Small_area.apply(lambda x:x[1:-1])
data.reset_index(drop=True,inplace=True)# inplace 和 赋值不能同时用
# data.reset_index(inplace=True)
# data.index=range(0,data.shape[0])
# print(data.describe())
# print(data.describe(include=['object']))


# import warnings
# warnings.filterwarnings("ignore")
# sns.pairplot(data,kind="reg")
# plt.show()


# 星级
# scores = data[['star_level','comment']].groupby(data['star_level']).count()
scores = data['star_level'].groupby(data['star_level']).count()
pie1 = Pie("星级", '各星级火锅店的数量占比', title_pos='center', width=800,height=800)
pie1.add(
    "",
    scores.index,
    scores.values,
    radius=[40, 75],
#    center=[50, 50],
    is_random=True,
#    radius=[30, 75],
    is_legend_show=False,
    is_label_show=False,
)
pie1.render('星级.html')


num_date = data.comment.groupby(data['star_level']).sum()
chart = Bar("各星级下的评论数分布","", title_color="#fff",title_pos="center", width=1200,height=600, background_color='#404a59')
chart.use_theme('dark')#使用主题
# chart.use_theme('light')
chart.add( '',num_date.index, num_date.values,mark_point=["min", "max","average"],is_legend_show=False,is_label_show=False)
# chart.print_echarts_options() # 该行只为了打印配置项，方便调试时使用
chart.render('各星级下的评论数分布.html')



num0 = data.consumption.groupby(data['Small_area']).mean().sort_values(ascending=False)
bar = Bar('各地区火锅均价')
# bar.use_theme('dark')#使用主题
# bar.use_theme('light')
bar.add( '直方图',num0.index, num0.values,mark_point=["min", "max","average"],mark_line=["min", "max","average"])
# chart.print_echarts_opti.ons() # 该行只为了打印配置项，方便调试时使用
line = Line()
line.add( '',num0.index, num0.values, is_fill=True, line_opacity=0.2,mark_point=["min", "max","average"],mark_line=["min", "max"],
          area_opacity=0.4, symbol=None)
# line.add( '折线图',num0.index, num0.values,is_smooth=True,mark_point=["min", "max","average"])# is_label_show=True
overlap = Overlap(width=1500, height=800)
overlap.add(bar)
overlap.add(line, yaxis_index=1, is_add_yaxis=True)
overlap.render('各地区火锅均价折线直方图.html')



num = data.consumption.groupby(data['Small_area']).mean().sort_values(ascending=False)[:20]
# 点赞数时间分布
chart = Bar("均价前二十各地区的分布","", title_color="#fff",title_pos="center", width=1500,height=600, background_color='#404a59')
chart.use_theme('dark')#使用主题
# chart.use_theme('light')
chart.add( '',num.index, num.values, is_fill=True, line_opacity=0.2,mark_point=["min", "max","average"],mark_line=["min", "max"],
          area_opacity=0.4, symbol=None)
# chart.print_echarts_options() # 该行只为了打印配置项，方便调试时使用
chart.render('均价前二十各地区的分布图.html')


num1= data.consumption.groupby(data['Small_area']).mean().sort_values()[:20]
# 点赞数时间分布
chart = Bar("均价后二十各地区的分布","", title_color="#fff",title_pos="center", width=1500,height=600, background_color='#404a59')
chart.use_theme('dark')#使用主题
# chart.use_theme('light')
chart.add( '',num1.index, num1.values, is_fill=True, line_opacity=0.2,
          area_opacity=0.4, symbol=None)
# chart.print_echarts_options() # 该行只为了打印配置项，方便调试时使用
chart.render('均价后二十各地区的分布图.html')


num2 = data.comment.groupby(data['consumption']).sum()
chart = Line("各价格下的评论数分布","", title_color="#fff",title_pos="center", width=1200,height=600, background_color='#404a59')
chart.use_theme('dark')#使用主题
# bar.use_theme('light')
chart.add( '',num2.index, num2.values, is_fill=True, line_opacity=0.2,
          area_opacity=0.4, symbol=None,mark_point=["min", "max","average"],xaxis_name = '价格',yaxis_name = '评论数')
# chart.print_echarts_options() # 该行只为了打印配置项，方便调试时使用
chart.render('各价格下的评论数分布.html')


chart = Scatter("口味·环境散点图" ,title_color="",title_pos="center")
chart.use_theme('light')
chart.add('', data.taste, data.environment, is_visualmap=True,
               xaxis_name = '环境',yaxis_name = '口味'
          )
chart.render('口味·环境散点图分布.html')


num3=data.region_name.value_counts()
funnel = Funnel('地区火锅数量漏斗图', title_pos='center')
funnel.add("", num3.index, num3.values, is_label_show=True, label_pos="inside", label_text_color="#fff", legend_pos="left", legend_orient="vertical")
funnel.render('地区火锅数量漏斗图分布.html')
# boxplot=Boxplot("1、2班考试成绩比较","箱线图",title_pos="center",width=1200,height=800)



data.dropna(inplace=True)
num4 = data.consumption.values
num5 = data.comment.values
boxplot = Boxplot('火锅价格·评论箱体图')
x_axis = ['价格','评论']
y_axis = [num4,num5]
yaxis = boxplot.prepare_data(y_axis)
boxplot.add("",x_axis,yaxis)
# boxplot.add("",num4.index,num4.values,is_label_show=True, label_pos="inside", label_text_color="#fff", legend_pos="left", legend_orient="vertical")
boxplot.render('火锅价格·评论箱体图分布.html')

# boxplot = Boxplot("最新评估与是否离职关系图", title_pos='center')
# x_axis = ['在职', '离职']
# y_axis = [df[df.left == 0].evaluation.values, df[df.left == 1].evaluation.values]
# boxplot.add("", x_axis, boxplot.prepare_data(y_axis))
# boxplot.render()

# from pyecharts import Bar, Pie, Grid
# #按照工作年限分别求离职人数和所有人数
# years_left_0 = df[df.left == 0].groupby('years_work')['left'].count()
# years_all = df.groupby('years_work')['left'].count()
# #分别计算离职人数和在职人数所占比例
# years_left0_rate = years_left_0 / years_all
# years_left1_rate = 1 - years_left0_rate
# attr = years_all.index
# bar = Bar("工作年限与是否离职的关系图", title_pos='10%')
# bar.add("离职", attr, years_left1_rate, is_stack=True)
# bar.add("在职", attr, years_left0_rate, is_stack=True, legend_pos="left" , legend_orient="vertical")
# #绘制圆环图
# pie = Pie("各工作年限所占百分比", title_pos='center')
# pie.add('', years_all.index, years_all, radius=[35, 60], label_text_color=None,
#         is_label_show=True, legend_orient="vertical", legend_pos="67%")
# grid = Grid(width=1200)
# grid.add(bar, grid_right="67%")
# grid.add(pie)
# grid.render()



num4 = data.consumption.values
boxplot = Boxplot('火锅价格箱体图')
x_axis = ['价格']
y_axis = [num4]
yaxis = boxplot.prepare_data(y_axis)
boxplot.add("",x_axis,yaxis)
# boxplot.add("",num4.index,num4.values,is_label_show=True, label_pos="inside", label_text_color="#fff", legend_pos="left", legend_orient="vertical")
boxplot.render('火锅价格箱体图分布.html')


num6 = data.consumption.groupby(data['shop']).mean().sort_values(ascending=False)
chart = Bar("所有火锅店的价格分布","", title_color="#fff",title_pos="center", width=1500,height=600, background_color='#404a59')
chart.use_theme('dark')#使用主题
# chart.use_theme('light')
chart.add( '',num6.index, num6.values,mark_point=["min", "max","average"],mark_line=["min", "max"])
# chart.print_echarts_options() # 该行只为了打印配置项，方便调试时使用
chart.render('所有火锅店的价格分布.html')


# data = data.corr()
# sns.heatmap(data)
# plt.show()


# 绘制直方图
data.consumption.plot(kind = 'hist', bins = 30, normed = True)
# 绘制核密度图
data.consumption.plot(kind = 'kde')
plt.savefig('质量分布图.png', dpi=600)
# 图形展现
plt.show()


sns.distplot(a = data.consumption, bins = 10, fit = stats.norm, norm_hist = True,
             hist_kws = {'color':'steelblue', 'edgecolor':'black'},
             kde_kws = {'color':'black', 'linestyle':'--', 'label':'核密度曲线'},
             fit_kws = {'color':'red', 'linestyle':':', 'label':'正态密度曲线'})
plt.savefig('质量分布图1.png', dpi=600)
plt.show()



data1=data.loc[data.consumption >= 200,:]
num7 = data1.consumption.groupby(data1['shop']).mean().sort_values(ascending=False)
chart = Bar("价格大于200的火锅店","", title_color="#fff",title_pos="center", width=1500,height=600, background_color='#404a59')
chart.use_theme('dark')#使用主题
# chart.use_theme('light')
chart.add( '',num7.index, num7.values,mark_point=["min", "max","average"],mark_line=["min", "max"])
# chart.print_echarts_options() # 该行只为了打印配置项，方便调试时使用
chart.render('价格大于200的火锅店.html')



# try:
# districts = ['运河区', '新华区', '泊头市', '任丘市', '黄骅市', '河间市', '沧县', '青县', '东光县', '海兴县', '盐山县', '肃宁县', '南皮县', '吴桥县', '献县', '孟村回族自治县']
# areas = [109.92, 109.47, 1006.5, 1023.0, 1544.7, 1333.0, 1104.0, 968.0, 730.0, 915.1, 796.0, 525.0, 794.0, 600.0, 1191.0, 387.0]
num8 = data.shop.groupby(data['region_name']).count()
map_1 = Map("各地区火锅数量", width=1600, height=900)
map_1.add("", num8.index, num8.values, maptype='重庆', is_visualmap=True, visual_range=[min(num0.values), max(num0.values)],
        visual_text_color='#000', is_map_symbol_show=False, is_label_show=True)
map_1.render('各地区火锅数量地图.html')


style = Style(title_color="#fff", title_pos="center",
              width=1200, height=600, background_color="#404a59")
map_2 = Map('各地区火锅数量', **style.init_style)
map_2.add("", num8.index, num8.values, visual_range=[0, 100],
        visual_text_color="#fff", type='heatmap',
        is_visualmap=True,is_piecewise = True, maptype='重庆'
        )
        # geo_cities_coords=geo_cities_coords)
map_2.render('各地区火锅数量热力图.html')


# except ValueError as e:
#     pass
    # e = str(e)
    # e = e.split("No coordinate is specified for ")[1]#获取不支持的城市名
    # for i in range(0,len(map_1)):
    #     if e in list(map_1[i]):
    #         del map_1[i]
    #         break



# # 点赞数每日内的时间分布
# num_time = datas.likes.groupby(datas['time']).count()
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
# datascore = datas.score.groupby(datas.dates).mean()
# chart = Line("评分时间分布")
# chart.use_theme('dark')
# chart.add('评分', datascore.index,
#           datascore.values,
#           line_width = 2
#           )
# chart.render('评分时间分布.html')
#
#
# """
# 评论分析
# """
#
# texts = ';'.join(datas.content.tolist())
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






# gender_map = {0: 'unknown', 1: 'male', 2: 'female'}
# data['gender'] = data['gender'].apply(lambda x: gender_map[x]) #映射
# data.sample(5) #随机选5行

# newdataset = dataset[-dataset['城区'].isin(['燕郊'])]



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