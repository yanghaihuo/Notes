# 导入第三方模块
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

os.chdir(r'C:\Users\Lenovo\Desktop\cdfangjia')
# 可视化的中文处理
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')
cd = pd.read_csv(r'C:\Users\Lenovo\Desktop\cdxinfang.csv',encoding='gbk')
# print(cd.shape)
# print(cd.head())
# 查看数据集是否存在缺失值
# print(cd.isnull())
# print(cd.apply(lambda x:np.sum(x.isnull())))
# print(cd.dtypes)
# print(any(cd.duplicated(subset='shop',inplace=True)))
# N = np.sum(cd.sale == 'None')
# Ratio = N/cd.shape[0]
# print(Ratio)

# N = np.sum(cd.price == '价格待定')
# Ratio = N/cd.shape[0]
# print(N)
# print(Ratio)

# print(any(cd.price.isnull()))
# cd=cd.loc[cd.price!='价格待定',:]
# cd=cd.drop_duplicates(subset='shop',inplace=True)

cd = cd[cd['states'].isin(['住宅在售','住宅售罄','住宅未开盘','住宅下期待开']) ]
cd.price = cd.price.str.extract('(.*?)元/平')
cd.dropna(subset=['price'],inplace=True)
cd['price'] = cd['price'].astype(int)

# cd=cd.loc[cd.sale!='None',:]
# print(cd.shape)
# cd.index=range(0,cd.shape[0])
# cd['new_sale']=cd.sale.str[:-1].astype('int')
# print(cd.new_sale)
# def func(x):
#     if x.find('万') != -1:
#         y = float(x[:-1]) * 10000
#     else :
#         y = float(x)
#     return y

# cd['new_estimate']=cd.estimate.apply(func)
# y=lambda x:float(x[:-1]) * 10000 if x.find('万') else float(x)
# cd['new_estimate']=cd.estimate.apply(y)
# z=lambda x:int(x)
# cd['new_estimate']=cd['new_estimate'].apply(z)

# print(cd.new_estimate)
# print(cd.new_estimate.dtype)

# 计算DataFrame每一列不重复元素的个数
# def func(x):
#     y=len(x.unique())
#     return y
# address_unique=cd.apply(func)
# print(address_unique)

# cd=cd.drop(['sale','estimate'],axis=1)

# print(cd.info())
# print(cd.describe())
# print(cd.describe(include=['object']))

y = cd['price']
# 直方图
min_price = y.min()
max_price = y.max()
z=np.arange(min_price,max_price+1000,1000)
plt.hist(y,
         bins = z,
         color = 'steelblue'
         )
# 设置坐标轴标签和标题
# percent = [str(round(i*100,2))+'%' for i in cd.price_zhuzai/cd.price_zhuzai.sum()]
# plt.xticks(x,label,rotation=30,fontsize='small')
plt.title('成都房价分布直方图')
plt.xlim(0,30000)
plt.ylim(1,50)
plt.xlabel('房价')
plt.ylabel('频数')
# 去除图形顶部边界和右边界的刻度
plt.tick_params(top='off', right='off')
# 图形显示
plt.savefig('成都房价分布直方图.png',dpi=600,bbox_inches = 'tight')
plt.show()



# 累积频率直方图
plt.hist(y,
         bins = np.arange(min_price,max_price+1000,1000),
         normed = True, # 设置为频率直方图
         cumulative = True, # 积累直方图
         color = 'steelblue', # 指定填充色
         )

# 添加水平参考线
plt.axhline(y = 0.1, color = 'yellow',linestyle = '--', linewidth = 2)
plt.axhline(y = 0.3, color = 'blue', linestyle = '--', linewidth = 2)
plt.axhline(y = 0.5, color = 'red', linestyle = '--', linewidth = 2)
plt.axhline(y = 0.7, color = 'green', linestyle = '--', linewidth = 2)

# 设置坐标轴标签和标题
plt.title('成都房价累积分布直方图')
plt.xlim(0,80000)
plt.xlabel('房价')
plt.ylabel('累积频率')
# 去除图形顶部边界和右边界的刻度
plt.tick_params(top='off', right='off')
# 图形显示
plt.savefig('成都房价累积分布直方图.png',dpi=600,bbox_inches = 'tight')
plt.show()





# 离散化分箱条形图
# 指定任意的切割点，将数据分段
price_cuts = pd.cut(y, bins =[min_price,5000,10000,15000,20000,max_price])
# 按照数据段，进行数据的统计，即频数统计
price_stats = price_cuts.value_counts()
x = range(len(price_stats))
# print(price_stats)
# 将索引用作绘图的刻度标签
label = price_stats.index
# 占比用于绘图的数值标签
percent = [str(round(i*100,2))+'%' for i in price_stats/price_stats.sum()]
# 绘图
plt.bar(x, # x轴数据
        price_stats, # y轴数据
        align = 'center', # 刻度居中对齐
        color='steelblue', # 填充色
        alpha = 0.8 # 透明度
       )
# 设置y轴的刻度范围
# plt.ylim(0,80)
# x轴刻度标签
plt.xticks(x,label,rotation=60,fontsize='small')

# 设置坐标轴标签和标题
plt.title('成都房价区间条形图')
plt.xlabel('房价区间')
plt.ylabel('频数')

# 去除图形顶部边界和右边界的刻度
plt.tick_params(top='off', right='off')

# 为每个条形图添加数值标签
for xh,yh,zh in zip(x,price_stats,percent):
    plt.text(xh, yh+1,'%s' %zh,ha='center')

# 显示图形
plt.savefig('成都房价区间条形图.png',dpi=600,bbox_inches = 'tight')
plt.show()





#饼图
# 指定任意的切割点，将数据分段
cuts = pd.cut(y, bins = [min_price,6000,8000,10000,12000,15000,18000,20000,max_price])
stats = cuts.value_counts()
label = stats.index
# 将横、纵坐标轴标准化处理，保证饼图是一个正圆，否则为椭圆
plt.axes(aspect='equal')
# 自定义颜色
colors=['#9999ff','#ff9999','#7777aa','#2442aa','#dd5555']
# 绘制饼图
plt.pie(stats,
        labels=label,
        colors = colors, # 设置颜色
        autopct='%.2f%%', # 设置百分比的格式，这里保留一位小数
        counterclock = False, # 设置为顺时针方向
        wedgeprops = {'linewidth': 1.5, 'edgecolor':'green'},# 设置饼图内外边界的属性值
        textprops = {'fontsize':12, 'color':'k'} # 设置文本标签的属性值
       )
# 添加图标题
plt.title('成都房价区间分布')
# 显示图形
plt.savefig('成都房价区间分布.png',dpi=600,bbox_inches = 'tight')
plt.show()



# from os import path

# d=path.dirname(__file__)
# print(d)
ls = []
categories = ['温江','龙泉驿','双流','新都','高新','锦江']
for cate in categories:
    # sub = app_info.loc[app_info.appcategory.apply(lambda x: x.find(cate) != -1), ['appname', 'love_new']]
    sub=cd.loc[cd.area==cate,['price', 'area']]
    sub = sub.sort_values(by=['price'],ascending=False)[:5]
    sub['type'] = cate
    ls.append(sub)
app_love_cat = pd.concat(ls)

# 为了让多张子图在一张图中完成，设置子图的位置
ax1 = plt.subplot2grid((3, 2), (0, 0))
ax2 = plt.subplot2grid((3, 2), (0, 1))
ax3 = plt.subplot2grid((3, 2), (1, 0))
ax4 = plt.subplot2grid((3, 2), (1, 1))
ax5 = plt.subplot2grid((3, 2), (2, 0))
ax6 = plt.subplot2grid((3, 2), (2, 1))

# 将图框存放起来，用于循环使用
axes = [ax1, ax2, ax3, ax4, ax5,ax6]
types = app_love_cat.type.unique()

# 循环的方式完成5张图的绘制
for i in range(6):
    # 准备绘图数据
    data = app_love_cat.loc[app_love_cat.type == types[i]]
    # 绘制条形图
    axes[i].bar(range(5), data.price, color='steelblue', alpha=0.7)
    # 设置图框大小
    gcf = plt.gcf()
    gcf.set_size_inches(8, 6)
    # 添加标题
    axes[i].set_title(types[i] + '前5的房价', size=9)
    # 删除各子图上、右和下的边界刻度标记
    axes[i].tick_params(top='off', bottom='off', right='off')

# 调整子图之间的水平间距和高度间距
plt.subplots_adjust(hspace=0.6, wspace=0.3)
plt.savefig('成都各地区前五房价.png',dpi=600,bbox_inches = 'tight')
# 显示图形
plt.show()



ls = []
categories = ['温江','龙泉驿','双流','新都','高新','锦江']
for cate in categories:
    # sub = app_info.loc[app_info.appcategory.apply(lambda x: x.find(cate) != -1), ['appname', 'love_new']]
    sub=cd.loc[cd.area==cate,['price', 'area']]
    sub = sub.sort_values(by=['price'])[:5]
    sub['type'] = cate
    ls.append(sub)
app_love_cat = pd.concat(ls)

# 为了让多张子图在一张图中完成，设置子图的位置
ax1 = plt.subplot2grid((3, 2), (0, 0))
ax2 = plt.subplot2grid((3, 2), (0, 1))
ax3 = plt.subplot2grid((3, 2), (1, 0))
ax4 = plt.subplot2grid((3, 2), (1, 1))
ax5 = plt.subplot2grid((3, 2), (2, 0))
ax6 = plt.subplot2grid((3, 2), (2, 1))

# 将图框存放起来，用于循环使用
axes = [ax1, ax2, ax3, ax4, ax5,ax6]
types = app_love_cat.type.unique()

# 循环的方式完成5张图的绘制
for i in range(6):
    # 准备绘图数据
    data = app_love_cat.loc[app_love_cat.type == types[i]]
    # 绘制条形图
    axes[i].bar(range(5), data.price, color='steelblue', alpha=0.7)
    # 设置图框大小
    gcf = plt.gcf()
    gcf.set_size_inches(8, 6)
    # 添加标题
    axes[i].set_title(types[i] + '后5的房价', size=9)
    # 删除各子图上、右和下的边界刻度标记
    axes[i].tick_params(top='off', bottom='off', right='off')

# 调整子图之间的水平间距和高度间距
plt.subplots_adjust(hspace=0.6, wspace=0.3)
plt.savefig('成都各地区后五房价.png',dpi=600,bbox_inches = 'tight')
# 显示图形
plt.show()


m=cd.groupby(['area']).mean()
# 绘图
plt.bar(np.arange(len(m.index)), m.price.values, align = 'center',color='steelblue', alpha = 0.8)
# 添加轴标签
plt.ylabel('均价')
# 添加标题
plt.title('成都各地区住宅均价')
# 添加刻度标签
plt.xticks(range(30),m.index,rotation=30)
for x,y in enumerate(m.price):
    plt.text(x,y+50,'%s' %round(y,2),ha='center')
plt.savefig('成都各地区住宅均价.png',dpi=600,bbox_inches = 'tight')
# 显示图形
plt.show()


n=cd.groupby(['area']).count()
# 绘图
plt.bar(np.arange(len(n.index)), n.price.values, align = 'center',color='steelblue', alpha = 0.8)
# 添加轴标签
plt.ylabel('频数')
# 添加标题
plt.title('成都各地区住宅套数')
# 添加刻度标签
plt.xticks(range(30),n.index,rotation=30)
plt.savefig('成都各地区住宅套数.png',dpi=600,bbox_inches = 'tight')
plt.show()