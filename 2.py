# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# f=open(r'C:\Users\Lenovo\PycharmProjects\pachonglianxi\163Music-master\music163\music163网易云音乐精彩评论.csv')
# data = pd.read_csv(f)



# # 1.垂直堆叠条形图
# # 导入模块
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
#
# # 导入数据
# f = open(r'C:\Users\Lenovo\Desktop\货运.csv',encoding='utf-8')
# data=pd.read_csv(f)
# print(data)
# # 中文乱码的处理
# plt.rcParams['font.sans-serif'] =['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# # 绘图
# plt.bar(np.arange(8), data.loc[0,:][1:], color = 'red', alpha = 0.8, label = '铁路', align = 'center')
# plt.bar(np.arange(8), data.loc[1,:][1:],  bottom = data.loc[0,:][1:], color = 'green', alpha = 0.8, label = '公路', align = 'center')
# plt.bar(np.arange(8), data.loc[2,:][1:],  bottom = data.loc[0,:][1:]+data.loc[1,:][1:], color = 'm', alpha = 0.8, label = '水运', align = 'center')
# plt.bar(np.arange(8), data.loc[3,:][1:],  bottom = data.loc[0,:][1:]+data.loc[1,:][1:]+data.loc[2,:][1:], color = 'black', alpha = 0.8, label = '民航', align = 'center')
# # 添加轴标签
# plt.xlabel('月份')
# plt.ylabel('货物量(万吨)')
# # 添加标题
# plt.title('2017年各月份物流运输量')
# # 添加刻度标签
# plt.xticks(np.arange(8),data.columns[1:])
# # 设置Y轴的刻度范围
# plt.ylim([0,500000])
#
# # 为每个条形图添加数值标签
# for x_t,y_t in enumerate(data.loc[0,:][1:]):
#     plt.text(x_t,y_t/2,'%sW' %(round(y_t/10000,2)),ha='center', color = 'white')
#
# for x_g,y_g in enumerate(data.loc[0,:][1:]+data.loc[1,:][1:]):
#     plt.text(x_g,y_g/2,'%sW' %(round(y_g/10000,2)),ha='center', color = 'white')
#
# for x_s,y_s in enumerate(data.loc[0,:][1:]+data.loc[1,:][1:]+data.loc[2,:][1:]):
#     plt.text(x_s,y_s-20000,'%sW' %(round(y_s/10000,2)),ha='center', color = 'white')
#
# # 显示图例
# plt.legend(loc='upper center', ncol=4)
# # 显示图形
# plt.show()



# 2.饼图
# # 导入第三方模块
# import matplotlib.pyplot as plt
#
# # 设置绘图的主题风格（不妨使用R中的ggplot分隔）
# plt.style.use('ggplot')
#
# # 构造数据
# edu = [0.2515, 0.3724, 0.3336, 0.0368, 0.0057]
# labels = ['中专', '大专', '本科', '硕士', '其他']
#
# explode = [0, 0.1, 0, 0, 0]  # 用于突出显示大专学历人群
# colors = ['#9999ff', '#ff9999', '#7777aa', '#2442aa', '#dd5555']  # 自定义颜色
#
# # 中文乱码和坐标轴负号的处理
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# # 将横、纵坐标轴标准化处理，保证饼图是一个正圆，否则为椭圆
# plt.axes(aspect='equal')
#
# # 控制x轴和y轴的范围
# plt.xlim(0, 4)
# plt.ylim(0, 4)
#
# # 绘制饼图
# plt.pie(x=edu,  # 绘图数据
#         explode=explode,  # 突出显示大专人群
#         labels=labels,  # 添加教育水平标签
#         colors=colors,  # 设置饼图的自定义填充色
#         autopct='%.1f%%',  # 设置百分比的格式，这里保留一位小数
#         pctdistance=0.8,  # 设置百分比标签与圆心的距离
#         labeldistance=1.15,  # 设置教育水平标签与圆心的距离
#         startangle=180,  # 设置饼图的初始角度
#         radius=1.5,  # 设置饼图的半径
#         counterclock=False,  # 是否逆时针，这里设置为顺时针方向
#         wedgeprops={'linewidth': 1.5, 'edgecolor': 'green'},  # 设置饼图内外边界的属性值
#         textprops={'fontsize': 12, 'color': 'k'},  # 设置文本标签的属性值
#         center=(1.8, 1.8),  # 设置饼图的原点
#         frame=1)  # 是否显示饼图的图框，这里设置显示
#
# # 删除x轴和y轴的刻度
# plt.xticks(())
# plt.yticks(())
#
# # 添加图标题
# plt.title('芝麻信用失信用户教育水平分布')
#
# # 显示图形
# plt.show()



# 导入模块
import pandas as pd
import matplotlib.pyplot as plt

# 设置绘图风格
plt.style.use('ggplot')
# 设置中文编码和负号的正常显示
plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

# 读取需要绘图的数据
f = open(r'C:\Users\Lenovo\Desktop\python_curveplot\wechart.csv')
article_reading = pd.read_csv(f)
# 取出8月份至9月28日的数据
sub_data = article_reading.loc[article_reading.date >= '2017-08-01' ,:]

# 设置图框的大小
fig = plt.figure(figsize=(10,6))
# 绘图
plt.plot(sub_data.date, # x轴数据
         sub_data.article_reading_cnts, # y轴数据
         linestyle = '-', # 折线类型
         linewidth = 2, # 折线宽度
         color = 'steelblue', # 折线颜色
         marker = 'o', # 点的形状
         markersize = 6, # 点的大小
         markeredgecolor='black', # 点的边框色
         markerfacecolor='brown') # 点的填充色

# 添加标题和坐标轴标签
plt.title('公众号每天阅读人数趋势图')
plt.xlabel('日期')
plt.ylabel('人数')

# 取出图框上边界和右边界的刻度
plt.tick_params(top = 'off', right = 'off')

# 为了避免x轴日期刻度标签的重叠或拥挤，设置x轴刻度自动展现，并且45度倾斜
fig.autofmt_xdate(rotation = 45)
# 显示图形
plt.show()



# 导入模块
import matplotlib as mpl

# 设置图框的大小
fig = plt.figure(figsize=(10,6))
# 绘图
plt.plot(sub_data.date, # x轴数据
         sub_data.article_reading_cnts, # y轴数据
         linestyle = '-', # 折线类型
         linewidth = 2, # 折线宽度
         color = 'steelblue', # 折线颜色
         marker = 'o', # 点的形状
         markersize = 6, # 点的大小
         markeredgecolor='black', # 点的边框色
         markerfacecolor='steelblue') # 点的填充色

# 添加标题和坐标轴标签
plt.title('公众号每天阅读人数趋势图')
plt.xlabel('日期')
plt.ylabel('人数')

# 取出图框上边界和右边界的刻度
plt.tick_params(top = 'off', right = 'off')

# 获取图的坐标信息
ax = plt.gca()
# 设置日期的显示格式
date_format = mpl.dates.DateFormatter("%Y-%m-%d")
ax.xaxis.set_major_formatter(date_format)

# 设置x轴显示多少个日期刻度
#xlocator = mpl.ticker.LinearLocator(10)
# 设置x轴每个刻度的间隔天数
xlocator = mpl.ticker.MultipleLocator(5)
ax.xaxis.set_major_locator(xlocator)

# 为了避免x轴日期刻度标签的重叠，设置x轴刻度自动展现，并且45度倾斜
fig.autofmt_xdate(rotation = 45)
# 显示图形
plt.show()



# 设置图框的大小
fig = plt.figure(figsize=(10,6))
# 绘图--阅读人数趋势
plt.plot(sub_data.date, # x轴数据
         sub_data.article_reading_cnts, # y轴数据
         linestyle = '-', # 折线类型
         linewidth = 2, # 折线宽度
         color = 'steelblue', # 折线颜色
         marker = 'o', # 点的形状
         markersize = 6, # 点的大小
         markeredgecolor='black', # 点的边框色
         markerfacecolor='steelblue', # 点的填充色
         label = '阅读人数') # 添加标签

# 绘图--阅读人次趋势
plt.plot(sub_data.date, # x轴数据
         sub_data.article_reading_times, # y轴数据
         linestyle = '-', # 折线类型
         linewidth = 2, # 折线宽度
         color = '#ff9999', # 折线颜色
         marker = 'o', # 点的形状
         markersize = 6, # 点的大小
         markeredgecolor='black', # 点的边框色
         markerfacecolor='#ff9999', # 点的填充色
         label = '阅读人次') # 添加标签

# 添加标题和坐标轴标签
plt.title('公众号每天阅读人数和人次趋势图')
plt.xlabel('日期')
plt.ylabel('人数')

# 取出图框上边界和右边界的刻度
plt.tick_params(top = 'off', right = 'off')

# 获取图的坐标信息
ax = plt.gca()
# 设置日期的显示格式
date_format = mpl.dates.DateFormatter('%m-%d')
ax.xaxis.set_major_formatter(date_format)

# 设置x轴显示多少个日期刻度
#xlocator = mpl.ticker.LinearLocator(10)
# 设置x轴每个刻度的间隔天数
xlocator = mpl.ticker.MultipleLocator(3)
ax.xaxis.set_major_locator(xlocator)

# 为了避免x轴日期刻度标签的重叠，设置x轴刻度自动展现，并且45度倾斜
fig.autofmt_xdate(rotation = 45)

# 显示图例
plt.legend()
# 显示图形
plt.show()