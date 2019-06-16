'柱状图 饼图 漏斗图 散点图 折线图'
'''
Function:
	分析拉勾网招聘数据
'''
import os
import pickle
from pyecharts import Bar
from pyecharts import Pie
from pyecharts import Funnel


# 柱状图(2维)
def DrawBar(title, data, savepath='./results'):
	if not os.path.exists(savepath):
		os.mkdir(savepath)
	bar = Bar(title)
	attrs = [i for i, j in data.items()]
	values = [j for i, j in data.items()]
	bar.add('', attrs, values, mark_point=["min", "max"], is_convert=True)
	bar.render(os.path.join(savepath, '%s.html' % title))


# 饼图
def DrawPie(title, data, savepath='./results'):
	if not os.path.exists(savepath):
		os.mkdir(savepath)
	pie = Pie(title, title_pos='center')
	attrs = [i for i, j in data.items()]
	values = [j for i, j in data.items()]
	pie.add('', attrs, values, is_label_show=True, radius=[30, 50], rosetype="radius", legend_pos="left", legend_orient="vertical")
	pie.render(os.path.join(savepath, '%s.html' % title))


# 漏斗图
def DrawFunnel(title, data, savepath='./results'):
	if not os.path.exists(savepath):
		os.mkdir(savepath)
	funnel = Funnel(title, title_pos='center')
	attrs = [i for i, j in data.items()]
	values = [j for i, j in data.items()]
	funnel.add("", attrs, values, is_label_show=True, label_pos="inside", label_text_color="#fff", legend_pos="left", legend_orient="vertical")
	funnel.render(os.path.join(savepath, '%s.html' % title))


if __name__ == '__main__':
	with open('info.pkl', 'rb') as f:
		data = pickle.load(f)
	'''
	salaryDict = {}
	for key, value in data.items():
		average = 0
		num = 0
		for v in value:
			num += 1
			salary = v[-1]
			salary = salary.split('-')
			try:
				tmp = (float(salary[0].replace('k', '').replace('K', '')) + float(salary[0].replace('k', '').replace('K', ''))) / 2
			except:
				continue
			average += tmp
		salaryDict[key] = average / num
	DrawBar(title='部分城市Python相关岗位平均薪资柱状图', data=salaryDict, savepath='./results')
	'''
	'''
	eduDict = {}
	for key, value in data.items():
		for v in value:
			edu = v[-3]
			if edu in eduDict:
				eduDict[edu] += 1
			else:
				eduDict[edu] = 1
	DrawPie(title='部分城市Python相关岗位应聘学历要求', data=eduDict, savepath='./results')
	'''
	'''
	companySizeDict = {}
	for key, value in data.items():
		for v in value:
			companySize = v[5]
			if companySize in companySizeDict:
				companySizeDict[companySize] += 1
			else:
				companySizeDict[companySize] = 1
	DrawFunnel(title='部分城市招聘Python相关岗位的公司规模', data=companySizeDict, savepath='./results')
	'''
	jobNatureDict = {}
	for key, value in data.items():
		for v in value:
			jobNature = v[6]
			if jobNature in jobNatureDict:
				jobNatureDict[jobNature] += 1
			else:
				jobNatureDict[jobNature] = 1
	DrawBar(title='部分城市Python相关岗位工作性质', data=jobNatureDict, savepath='./results')

	industryFieldDict = {}
	for key, value in data.items():
		for v in value:
			industryField = v[4]
			try:
				industryFields = industryField.split(',')
			except:
				continue
			if len(industryFields) == 1:
				industryFields = industryField.split(' ')
			if len(industryFields) == 1:
				industryFields = industryField.split('、')
			for industryField in industryFields:
				industryField = industryField.strip(' ')
				if industryField in industryFieldDict:
					industryFieldDict[industryField] += 1
				else:
					industryFieldDict[industryField] = 1
	DrawPie(title='部分城市Python相关岗位工作领域', data=industryFieldDict, savepath='./results')



import os
import json
from pyecharts import Bar
from pyecharts import Pie
from pyecharts import Funnel
from pyecharts import Scatter


# 柱状图(2维)
def DrawBar(title, data, savepath='./results'):
	if not os.path.exists(savepath):
		os.mkdir(savepath)
	bar = Bar(title)
	attrs = [i for i, j in data.items()]
	values = [j for i, j in data.items()]
	bar.add('', attrs, values, mark_point=["min", "max"])
	bar.render(os.path.join(savepath, '%s.html' % title))


# 饼图
def DrawPie(title, data, savepath='./results'):
	if not os.path.exists(savepath):
		os.mkdir(savepath)
	pie = Pie(title, title_pos='center')
	attrs = [i for i, j in data.items()]
	values = [j for i, j in data.items()]
	pie.add('', attrs, values, is_label_show=True, radius=[30, 50], rosetype="radius", legend_pos="left", legend_orient="vertical")
	pie.render(os.path.join(savepath, '%s.html' % title))


# 漏斗图
def DrawFunnel(title, data, savepath='./results'):
	if not os.path.exists(savepath):
		os.mkdir(savepath)
	funnel = Funnel(title, title_pos='center')
	attrs = [i for i, j in data.items()]
	values = [j for i, j in data.items()]
	funnel.add("", attrs, values, is_label_show=True, label_pos="inside", label_text_color="#fff", legend_pos="left", legend_orient="vertical")
	funnel.render(os.path.join(savepath, '%s.html' % title))


# 散点图
def DrawScatter(title, data, savepath='./results'):
	if not os.path.exists(savepath):
		os.mkdir(savepath)
	scatter = Scatter(title)
	attrs = [i for i, j in data.items()]
	values = [j for i, j in data.items()]
	scatter.add('', attrs, values, is_visualmap=True)
	scatter.render(os.path.join(savepath, '%s.html' % title))


if __name__ == '__main__':
	with open('anjuke.json', 'r') as f:
		data = json.loads(f.read())
	'''
	prices = {'0到2000/月': 0, '2000到4000/月': 0, '4000到6000/月': 0, '6000到8000/月': 0, '8000到10000/月': 0, '大于10000/月': 0}
	for d in data:
		price = int(d.get('price'))
		if price in range(0, 2000):
			prices['0到2000/月'] += 1
		elif price in range(2000, 4000):
			prices['2000到4000/月'] += 1
		elif price in range(4000, 6000):
			prices['4000到6000/月'] += 1
		elif price in range(6000, 8000):
			prices['6000到8000/月'] += 1
		elif price in range(8000, 10000):
			prices['8000到10000/月'] += 1
		else:
			prices['大于10000/月'] += 1
	DrawBar(title='上海租房月租金分布柱状图', data=prices, savepath='./results')
	DrawPie(title='上海租房月租金分布饼图', data=prices, savepath='./results')
	'''
	'''
	prices_avg = {'0到50/月': 0, '50到100/月': 0, '100到150/月': 0, '150到200/月': 0, '大于200/月': 0}
	for d in data:
		price = int(float(d.get('price')) / float(d.get('area')))
		if price in range(0, 50):
			prices_avg['0到50/月'] += 1
		elif price in range(50, 100):
			prices_avg['50到100/月'] += 1
		elif price in range(100, 150):
			prices_avg['100到150/月'] += 1
		elif price in range(150, 200):
			prices_avg['150到200/月'] += 1
		else:
			prices_avg['大于200/月'] += 1
	DrawPie(title='上海租房(单位面积)月租金分布饼图', data=prices_avg, savepath='./results')
	'''
	'''
	towards_dict = {}
	for d in data:
		towards = d.get('towards')
		if towards in towards_dict:
			towards_dict[towards] += 1
		else:
			towards_dict[towards] = 1
	DrawFunnel(title='上海出租住房朝向漏斗图', data=towards_dict, savepath='./results')
	'''
	'''
	building_type_dict = {}
	for d in data:
		building_type = d.get('building_type')
		if building_type in building_type_dict:
			building_type_dict[building_type] += 1
		else:
			building_type_dict[building_type] = 1
	DrawBar(title='上海出租住房类型柱状图', data=building_type_dict, savepath='./results')
	'''



'''
Function:
	简单分析一下爬到的数据
作者:
	Charles
公众号:
	Charles的皮卡丘
'''
import os
import pickle
from pyecharts import Pie
from pyecharts import Bar
from pyecharts import Line
from pyecharts import Funnel
from pyecharts import Scatter
from pyecharts import Overlap


# 饼图
def DrawPie(title, data, savepath='./results'):
	if not os.path.exists(savepath):
		os.mkdir(savepath)
	pie = Pie(title, title_pos='center')
	attrs = [i for i, j in data.items()]
	values = [j for i, j in data.items()]
	pie.add('', attrs, values, is_label_show=True, radius=[30, 50], rosetype="radius", legend_pos="left", legend_orient="vertical")
	pie.render(os.path.join(savepath, '%s.html' % title))


# 柱状图(2维)
def DrawBar(title, data, savepath='./results'):
	if not os.path.exists(savepath):
		os.mkdir(savepath)
	bar = Bar(title)
	attrs = [i for i, j in data.items()]
	values = [j for i, j in data.items()]
	bar.add('', attrs, values, mark_point=["min", "max"], xaxis_rotate=30, yaxis_rotate=30)
	bar.render(os.path.join(savepath, '%s.html' % title))


# 散点图
def DrawScatter(title, data, savepath='./results'):
	if not os.path.exists(savepath):
		os.mkdir(savepath)
	scatter = Scatter(title)
	attrs = [i for i, j in data.items()]
	values = [j for i, j in data.items()]
	scatter.add('', attrs, values, is_visualmap=True)
	scatter.render(os.path.join(savepath, '%s.html' % title))


# 漏斗图
def DrawFunnel(title, data, savepath='./results'):
	if not os.path.exists(savepath):
		os.mkdir(savepath)
	funnel = Funnel(title, title_pos='center')
	attrs = [i for i, j in data.items()]
	values = [j for i, j in data.items()]
	funnel.add("", attrs, values, is_label_show=True, label_pos="inside", label_text_color="#fff", legend_pos="left", legend_orient="vertical")
	funnel.render(os.path.join(savepath, '%s.html' % title))


# 折线图(2维)
def DrawLine(title, data, savepath='./results'):
	if not os.path.exists(savepath):
		os.mkdir(savepath)
	line = Line(title)
	attrs = [i for i, j in data.items()]
	values = [j for i, j in data.items()]
	line.add('', attrs, values, is_smooth=True, mark_point=["max", "min"])
	line.render(os.path.join(savepath, '%s.html' % title))


# 折线图加柱状图
def DrawBarLine(title, data, savepath='./results'):
	if not os.path.exists(savepath):
		os.mkdir(savepath)
	bar = Bar(title)
	attrs = [i for i, j in data.items()]
	values = [j[1] for i, j in data.items()]
	bar.add('', attrs, values, xaxis_interval=0, yaxis_interval=50, xaxis_rotate=30, yaxis_rotate=30, yaxis_min=1960, yaxis_max=2020, mark_point=["max", "min"])
	line = Line(title)
	attrs = [i for i, j in data.items()]
	values = [j[0] for i, j in data.items()]
	line.add('', attrs, values, is_smooth=True, mark_point=["max", "min"])
	overlap = Overlap(width=1200, height=600)
	overlap.add(bar)
	overlap.add(line, yaxis_index=1, is_add_yaxis=True)
	overlap.render(os.path.join(savepath, '%s.html' % title))


if __name__ == '__main__':
	with open('step1.pkl', 'rb') as f:
		data1 = pickle.load(f)
	with open('step2.pkl', 'rb') as f:
		data2 = pickle.load(f)
	'''
	# 评分TOP10
	data = {'None': 0.}
	for key, value in data1.items():
		try:
			float(value[0])
		except:
			continue
		if float(value[0]) > min(list(data.values())):
			if len(list(data.values())) == 10:
				data.pop(list(data.keys())[list(data.values()).index(min(list(data.values())))])
			data[key] = float(value[0])
	data_used = {}
	for key, value in data.items():
		if key == '双筒望远镜':
			data_used[key] = [value, 1997]
		elif key == '向阳理发店':
			data_used[key] = [value, 2001]
		elif key == '东寺街西寺巷':
			data_used[key] = [value, 2003]
		else:
			data_used[key] = [value, int(data2.get(key)[-2].split('-')[0])]
	DrawBarLine(title='豆瓣国产电视剧TOP10及其上映时间', data=data_used, savepath='./results')
	'''
	'''
	# 评分BOTTOM10
	data = {'None': 20.}
	for key, value in data1.items():
		try:
			float(value[0])
		except:
			continue
		if float(value[0]) < max(list(data.values())):
			if len(list(data.values())) == 10:
				data.pop(list(data.keys())[list(data.values()).index(max(list(data.values())))])
			data[key] = float(value[0])
	data_used = {}
	for key, value in data.items():
		if key == '新生活大爆炸':
			data_used[key] = [value, 2010]
		elif key == '敌后便衣队传奇':
			data_used[key] = [value, 2012]
		elif key == 'K时代':
			data_used[key] = [value, 2010]
		elif key == '游击英雄':
			data_used[key] = [value, 2015]
		elif key == '满山打鬼子':
			data_used[key] = [value, 2014]
		elif key == '刺蝶':
			data_used[key] = [value, 2014]
		elif key == '公主出山':
			data_used[key] = [value, 2012]
		elif key == '军统枪口下的女人':
			data_used[key] = [value, 2011]
		elif key == '不一样的美男子2':
			data_used[key] = [value, 2017]
		else:
			data_used[key] = [value, int(data2.get(key)[-2].split('-')[0])]
	DrawBarLine(title='豆瓣国产电视剧BOTTOM10及其上映时间', data=data_used, savepath='./results')4
	'''
	'''
	# 电视剧类型
	data = {}
	for key, value in data2.items():
		genres = value[2]
		for genre in genres:
			if genre in data:
				data[genre] += 1
			else:
				data[genre] = 1
	DrawPie(title='国产电视剧类型分布', data=data, savepath='./results')
	'''
	'''
	# 演员频次
	data = {}
	for key, value in data2.items():
		actors = value[0]
		if actors is None:
			continue
		for actor in actors:
			if actor in data:
				data[actor] += 1
			else:
				data[actor] = 1
	data_used = {'None': -1}
	for key, value in data.items():
		if float(value) > min(list(data_used.values())):
			if len(list(data_used.values())) == 20:
				data_used.pop(list(data_used.keys())[list(data_used.values()).index(min(list(data_used.values())))])
			data_used[key] = float(value)
	DrawBar(title='演员出演电视剧频次统计', data=data_used, savepath='./results')
	'''
	'''
	# 导演频次
	data = {}
	for key, value in data2.items():
		actors = value[1]
		if actors is None:
			continue
		for actor in actors:
			if actor in data:
				data[actor] += 1
			else:
				data[actor] = 1
	data_used = {'None': -1}
	for key, value in data.items():
		if float(value) > min(list(data_used.values())):
			if len(list(data_used.values())) == 20:
				data_used.pop(list(data_used.keys())[list(data_used.values()).index(min(list(data_used.values())))])
			data_used[key] = float(value)
	DrawBar(title='导演电视剧者频次统计', data=data_used, savepath='./results')
	'''