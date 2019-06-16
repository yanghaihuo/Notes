import matplotlib # 注意这个也要import一次
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize # import figsize

plt.rcParams['figure.figsize'] = (8.0, 4.0) # 设置figure_size尺寸
plt.rcParams['image.interpolation'] = 'nearest' # 设置 interpolation style
plt.rcParams['image.cmap'] = 'gray' # 设置 颜色 style

#figsize(12.5, 4) # 设置 figsize
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率
# 默认的像素：[6.0,4.0]，分辨率为100，图片尺寸为 600&400
# 指定dpi=200，图片尺寸为 1200*800
# 指定dpi=300，图片尺寸为 1800*1200
# 设置figsize可以在不改变分辨率情况下改变比例

#figsize(12.5, 4) # 设置 figsize
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率
# 默认的像素：[6.0,4.0]，分辨率为100，图片尺寸为 600&400
# 指定dpi=200，图片尺寸为 1200*800
# 指定dpi=300，图片尺寸为 1800*1200
# 设置figsize可以在不改变分辨率情况下改变比例

myfont = matplotlib.font_manager.FontProperties(fname=r'C:/Windows/Fonts/msyh.ttf') # 这一行
plt.plot((1,2,3),(4,3,-1))
plt.xlabel(u'横坐标',  fontproperties=myfont)
plt.ylabel(u'纵坐标',  fontproperties=myfont)
#plt.show()
plt.savefig('plot123_2.png', dpi=300) #指定分辨率保存

# brand_ex_1 = data[~data['品牌'].isin([1])]

# 批量转为数值型
# df = df.apply(pd.to_numeric,errors='ignore')
# con = df[col].str.contains('万$')
# df.loc[con,col] = pd.to_numeric(df.loc[con,col].str.replace('万','')) * 10000
# df[col] = pd.to_numeric(df[col])

'''
需要归一化的模型有
神经网络，标准差归一化
支持向量机，标准差归一化
线性回归，可以用梯度下降法求解，需要标准差归一化
PCA
LDA
聚类算法基本都需要
K近邻，线性归一化，归一到[0,1]区间内。 
逻辑回归

不需要归一化的模型：
决策树：  每次筛选都只考虑一个变量，不考虑变量之间的相关性，所以不需要归一化。
随机森林：不需要归一化，mtry为变量个数的均方根。
朴素贝叶斯

需要正则化的模型：
Lasso
Elastic Net
'''

from os import listdir
a=listdir(r'F:\360MoveData\Users\Administrator\Desktop\keshihua')

# DataFrame.sample(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None)
# replace参数的意思是：是否允许抽样值重复

import seaborn as sns
#用图表展现出各变量之间的相关性
#corr = df.corr()

