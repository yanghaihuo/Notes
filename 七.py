# import pymysql
# db = pymysql.connect(host='127.0.0.1', user='root', password='774110919', port=3306, db='maoyan')
# cursor = db.cursor()
# sql = 'CREATE TABLE IF NOT EXISTS films (name VARCHAR(255) NOT NULL, type VARCHAR(255) NOT NULL, country VARCHAR(255) NOT NULL, length VARCHAR(255) NOT NULL, released VARCHAR(255) NOT NULL, score VARCHAR(255) NOT NULL, people INT NOT NULL, box_office BIGINT NOT NULL, PRIMARY KEY (name))'
# cursor.execute(sql)
# db.close()


from pyecharts import Bar,Pie,TreeMap
import pandas as pd
import numpy as np
import pymysql

conn = pymysql.connect(host='localhost', user='root', password='774110919', port=3306, db='maoyan', charset='utf8mb4')
cursor = conn.cursor()
sql = "select * from films"
db = pd.read_sql(sql, conn)
df = db.sort_values(by="box_office", ascending=False)
dom = df[['name', 'box_office']]

attr = np.array(dom['name'][0:10])
v1 = np.array(dom['box_office'][0:10])
attr = ["{}".format(i.replace('：无限战争', '')) for i in attr]
v1 = ["{}".format(float('%.2f' % (float(i) / 100000000))) for i in v1]

bar = Bar("2018年电影票房TOP10(亿元)", title_pos='center', title_top='18', width=800, height=400)
bar.add("", attr, v1, is_convert=True, xaxis_min=10, yaxis_label_textsize=12, is_yaxis_boundarygap=True, yaxis_interval=0, is_label_show=True, is_legend_show=False, label_pos='right', is_yaxis_inverse=True, is_splitline_show=False)
bar.render("2018年电影票房TOP10.html")


conn = pymysql.connect(host='localhost', user='root', password='774110919', port=3306, db='maoyan', charset='utf8mb4')
cursor = conn.cursor()
sql = "select * from films"
db = pd.read_sql(sql, conn)
list1 = []
for i in db['country']:
    type1 = i.split(',')[0]
    if type1 in ['中国大陆', '中国香港']:
        type1 = '中国'
    else:
        type1 = '外国'
    list1.append(type1)
db['country_type'] = list1

country_type_message = db.groupby(['country_type'])
country_type_com = country_type_message['box_office'].agg(['sum'])
country_type_com.reset_index(inplace=True)
country_type_com_last = country_type_com.sort_index()

attr = country_type_com_last['country_type']
v1 = np.array(country_type_com_last['sum'])
v1 = ["{}".format(float('%.2f' % (float(i) / 100000000))) for i in v1]

pie = Pie("2018年中外电影票房对比(亿元)", title_pos='center')
pie.add("", attr, v1, radius=[40, 75], label_pos='right', label_text_color=None, is_label_show=True, legend_orient="vertical", legend_pos="left",label_formatter='{c}')
pie.render("2018年中外电影票房对比(亿元).html")


conn = pymysql.connect(host='localhost', user='root', password='774110919', port=3306, db='maoyan', charset='utf8mb4')
cursor = conn.cursor()
sql = "select * from films"
db = pd.read_sql(sql, conn)
list1 = []
for i in db['country']:
    place = i.split(',')[0]
    list1.append(place)
db['location'] = list1

place_message = db.groupby(['location'])
place_com = place_message['location'].agg(['count'])
place_com.reset_index(inplace=True)
place_com_last = place_com.sort_index()
dom = place_com_last.sort_values('count', ascending=False)[0:10]

attr = dom['location']
v1 = dom['count']

bar = Bar("2018年各国家电影数量TOP10", title_pos='center', title_top='18', width=800, height=400)
bar.add("", attr, v1, is_stack=True, is_label_show=True)
bar.render("2018年各国家电影数量TOP10.html")


def my_difference(a, b, c):
    rate = (a - b) / c
    return rate

conn = pymysql.connect(host='localhost', user='root', password='774110919', port=3306, db='maoyan', charset='utf8mb4')
cursor = conn.cursor()
sql = "select * from films"
a = pd.read_sql(sql, conn)
a['sort_num_money'] = a['box_office'].rank(ascending=0, method='dense')
a['sort_num_score'] = a['score'].rank(ascending=0, method='dense')
a['value'] = a.apply(lambda row: my_difference(row['sort_num_money'], row['sort_num_score'], len(a.index)), axis=1)
df = a.sort_values(by="value", ascending=True)[0:9]


v1 = ["{}".format('%.2f' % abs(i * 100)) for i in df['value']]
attr = np.array(df['name'])
attr = ["{}".format(i.replace('：无限战争', '')) for i in attr]

bar = Bar("2018年叫座不叫好电影TOP10", title_pos='center', title_top='18', width=800, height=400)
bar.add("", attr, v1, is_convert=True, xaxis_min=0, xaxis_max=4, yaxis_label_textsize=12, is_yaxis_boundarygap=True, yaxis_interval=0, is_label_show=True, is_legend_show=False, label_pos='right', is_yaxis_inverse=True, is_splitline_show=False)
bar.render("2018年叫座不叫好电影TOP100.html")


def my_sum(a, b, c):
    rate = (a + b) / c
    result = float('%.4f' % rate)
    return result

conn = pymysql.connect(host='localhost', user='root', password='774110919', port=3306, db='maoyan', charset='utf8mb4')
cursor = conn.cursor()
sql = "select * from films"
db = pd.read_sql(sql, conn)
db['sort_num_money'] = db['box_office'].rank(ascending=0, method='dense')
db['sort_num_score'] = db['score'].rank(ascending=0, method='dense')
db['value'] = db.apply(lambda row: my_sum(row['sort_num_money'], row['sort_num_score'], len(db.index)), axis=1)
df = db.sort_values(by="value", ascending=True)[0:10]

v1 = ["{}".format('%.2f' % ((1-i) * 100)) for i in df['value']]
attr = np.array(df['name'])
attr = ["{}".format(i.replace('：无限战争', '').replace('：全面瓦解', '')) for i in attr]

bar = Bar("2018年电影名利双收TOP10(%)", title_pos='center', title_top='18', width=800, height=400)
bar.add("", attr, v1, is_convert=True, xaxis_min=85, xaxis_max=100, yaxis_label_textsize=12, is_yaxis_boundarygap=True, yaxis_interval=0, is_label_show=True, is_legend_show=False, label_pos='right', is_yaxis_inverse=True, is_splitline_show=False)
bar.render("2018年电影名利双收TOP10.html")


conn = pymysql.connect(host='localhost', user='root', password='774110919', port=3306, db='maoyan', charset='utf8mb4')
cursor = conn.cursor()
sql = "select * from films"
db = pd.read_sql(sql, conn)
df = db.sort_values(by="released", ascending=False)
dom = df[['name', 'released']]
list1 = []
for i in dom['released']:
    time = i.split('-')[1]
    list1.append(time)
db['month'] = list1

month_message = db.groupby(['month'])
month_com = month_message['box_office'].agg(['sum'])
month_com.reset_index(inplace=True)
month_com_last = month_com.sort_index()

attr = ["{}".format(str(i) + '月') for i in range(1, 12)]
v1 = np.array(month_com_last['sum'])

v1 = ["{}".format(float('%.2f' % (float(i) / 100000000))) for i in v1]
bar = Bar("2018年每月电影票房(亿元)", title_pos='center', title_top='18', width=800, height=400)
bar.add("", attr, v1, is_stack=True, is_label_show=True)
bar.render("2018年每月电影票房(亿元).html")


conn = pymysql.connect(host='localhost', user='root', password='774110919', port=3306, db='maoyan', charset='utf8mb4')
cursor = conn.cursor()
sql = "select * from films"
db = pd.read_sql(sql, conn)
df = db.sort_values(by="released", ascending=False)
dom = df[['name', 'released']]
list1 = []
for i in dom['released']:
    place = i.split('-')[1]
    list1.append(place)
db['month'] = list1

month_message = db.groupby(['month'])
month_com = month_message['month'].agg(['count'])
month_com.reset_index(inplace=True)
month_com_last = month_com.sort_index()

attr = ["{}".format(str(i) + '月') for i in range(1, 12)]
v1 = np.array(month_com_last['count'])
v1 = ["{}".format(i) for i in v1]

bar = Bar("2018年每月上映电影数量", title_pos='center', title_top='18', width=800, height=400)
bar.add("", attr, v1, is_stack=True, yaxis_max=40, is_label_show=True)
bar.render("2018年每月上映电影数量.html")


conn = pymysql.connect(host='localhost', user='root', password='774110919', port=3306, db='maoyan', charset='utf8mb4')
cursor = conn.cursor()
sql = "select * from films"
db = pd.read_sql(sql, conn)
df = db.sort_values(by="people", ascending=False)
dom = df[['name', 'people']]

attr = np.array(dom['name'][0:10])
v1 = np.array(dom['people'][0:10])
attr = ["{}".format(i.replace('：无限战争', '')) for i in attr]
v1 = ["{}".format(float('%.2f' % (float(i) / 10000))) for i in v1]

bar = Bar("2018年电影人气TOP10(万人)", title_pos='center', title_top='18', width=800, height=400)
bar.add("", attr, v1, is_convert=True, xaxis_min=10, yaxis_label_textsize=12, is_yaxis_boundarygap=True, yaxis_interval=0, is_label_show=True, is_legend_show=False, label_pos='right', is_yaxis_inverse=True, is_splitline_show=False)
bar.render("2018年电影人气TOP10.html")


conn = pymysql.connect(host='localhost', user='root', password='774110919', port=3306, db='maoyan', charset='utf8mb4')
cursor = conn.cursor()
sql = "select * from films"
db = pd.read_sql(sql, conn)
df = db.sort_values(by="people", ascending=False)
dom = df[['name', 'people']]

attr = np.array(dom['name'][0:10])
v1 = np.array(dom['people'][0:10])
attr = ["{}".format(i.replace('：无限战争', '')) for i in attr]
v1 = ["{}".format(float('%.2f' % (float(i) / 10000))) for i in v1]

bar = Bar("2018年电影人气TOP10(万人)", title_pos='center', title_top='18', width=800, height=400)
bar.add("", attr, v1, is_convert=True, xaxis_min=10, yaxis_label_textsize=12, is_yaxis_boundarygap=True, yaxis_interval=0, is_label_show=True, is_legend_show=False, label_pos='right', is_yaxis_inverse=True, is_splitline_show=False)
bar.render("2018年电影人气TOP10.html")


conn = pymysql.connect(host='localhost', user='root', password='774110919', port=3306, db='maoyan', charset='utf8mb4')
cursor = conn.cursor()
sql = "select * from films"
db = pd.read_sql(sql, conn)
df = db.sort_values(by="score", ascending=False)
dom = df[['name', 'score']]

v1 = []
for i in dom['score'][0:10]:
    number = float(i.replace('分', ''))
    v1.append(number)
attr = np.array(dom['name'][0:10])
attr = ["{}".format(i.replace('：致命守护者', '')) for i in attr]

bar = Bar("2018年电影评分TOP10", title_pos='center', title_top='18', width=800, height=400)
bar.add("", attr, v1, is_convert=True, xaxis_min=8, xaxis_max=9.8, yaxis_label_textsize=10, is_yaxis_boundarygap=True, yaxis_interval=0, is_label_show=True, is_legend_show=False, label_pos='right', is_yaxis_inverse=True, is_splitline_show=False)
bar.render("2018年电影评分TOP10.html")


conn = pymysql.connect(host='localhost', user='root', password='774110919', port=3306, db='maoyan', charset='utf8mb4')
cursor = conn.cursor()
sql = "select * from films"
db = pd.read_sql(sql, conn)

dom1 = []
for i in db['type']:
    type1 = i.split(',')
    for j in range(len(type1)):
        if type1[j] in dom1:
            continue
        else:
            dom1.append(type1[j])

dom2 = []
for item in dom1:
    num = 0
    for i in db['type']:
        type2 = i.split(',')
        for j in range(len(type2)):
            if type2[j] == item:
                num += 1
            else:
                continue
    dom2.append(num)

def message():
    for k in range(len(dom2)):
        data = {}
        data['name'] = dom1[k] + ' ' + str(dom2[k])
        data['value'] = dom2[k]
        yield data

data1 = message()
dom3 = []
for item in data1:
    dom3.append(item)

treemap = TreeMap("2018年电影类型分布图", title_pos='center', title_top='5', width=800, height=400)
treemap.add('2018年电影类型分布', dom3, is_label_show=True, label_pos='inside', is_legend_show=False)
treemap.render('2018年电影类型分布图.html')

