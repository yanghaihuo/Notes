import requests
import psycopg2
import random
import time
import json

count = 0
url = 'https://www.lagou.com/jobs/positionAjax.json?needAddtionalResult=false'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36',
    'Cookie': '你的cookie值',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Connection': 'keep-alive',
    'Host': 'www.lagou.com',
    'Origin': 'https://www.lagou.com',
    'Referer': 'https://www.lagou.com/jobs/list_%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90?labelWords=sug&fromSearch=true&suginput=shuju'
}

# 连接数据库
db = psycopg2.connect(database="lagou_job", user="postgres", password="774110919", host="127.0.0.1", port="5432")


def add_Postgresql(id, job_title, job_salary, job_city, job_experience, job_education, company_name, company_type, company_status, company_people, job_tips, job_welfare):
    # 将数据写入数据库中
    try:
        cursor = db.cursor()
        sql = "insert into job (id, job_title, job_salary, job_city, job_experience, job_education, company_name, company_type, company_status, company_people, job_tips, job_welfare) values ('%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s')" % (int(id), job_title, job_salary, job_city, job_experience, job_education, company_name, company_type, company_status, company_people, job_tips, job_welfare);
        print(sql)
        cursor.execute(sql);
        print(cursor.lastrowid)
        db.commit()
    except Exception as e:
        print(e)
        db.rollback()


def get_message():
    for i in range(1, 31):
        print('第' + str(i) + '页')
        time.sleep(random.randint(10, 20))
        data = {
            'first': 'false',
            'pn': i,
            'kd': '数据挖掘'
        }
        response = requests.post(url=url, data=data, headers=headers)
        result = json.loads(response.text)
        job_messages = result['content']['positionResult']['result']
        job_urls = result['content']['hrInfoMap'].keys()
        for k in job_urls:
            url_1 = 'https://www.lagou.com/jobs/' + k + '.html'
            print(url_1)
            with open('job_urls.csv', 'a+', encoding='utf-8-sig') as f:
                f.write(url_1 + '\n')
        for job in job_messages:
            global count
            count += 1
            # 岗位名称
            job_title = job['positionName']
            print(job_title)
            # 岗位薪水
            job_salary = job['salary']
            print(job_salary)
            # 岗位地点
            job_city = job['city']
            print(job_city)
            # 岗位经验
            job_experience = job['workYear']
            print(job_experience)
            # 岗位学历
            job_education = job['education']
            print(job_education)
            # 公司名称
            company_name = job['companyShortName']
            print(company_name)
            # 公司类型
            company_type = job['industryField']
            print(company_type)
            # 公司状态
            company_status = job['financeStage']
            print(company_status)
            # 公司规模
            company_people = job['companySize']
            print(company_people)
            # 工作技能
            if len(job['positionLables']) > 0:
                job_tips = ','.join(job['positionLables'])
            else:
                job_tips = 'None'
            print(job_tips)
            # 工作福利
            job_welfare = job['positionAdvantage']
            print(job_welfare + '\n\n')
            add_Postgresql(count, job_title, job_salary, job_city, job_experience, job_education, company_name, company_type, company_status, company_people, job_tips, job_welfare)


if __name__ == '__main__':
    get_message()



from pyecharts import TreeMap
import pandas as pd
import psycopg2

conn = psycopg2.connect(database="lagou_job", user="postgres", password="774110919", host="127.0.0.1", port="5432")
cursor = conn.cursor()
sql = "select * from job"
df = pd.read_sql(sql, conn)

dom1 = []
for i in df['job_tips']:
    type1 = i.split(',')
    for j in range(len(type1)):
        if type1[j] in ['None', '大数据', '金融', '电商', '广告营销', '移动互联网']:
            continue
        else:
            if type1[j] in dom1:
                continue
            else:
                dom1.append(type1[j])
print(dom1)
dom2 = []
for item in dom1:
    num = 0
    for i in df['job_tips']:
        type2 = i.split(',')
        for j in range(len(type2)):
            if type2[j] in ['None', '大数据', '金融', '电商', '广告营销', '移动互联网']:
                continue
            else:
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

treemap = TreeMap("拉勾网数据挖掘岗—技能图", title_pos='center', title_top='5', width=800, height=400)
treemap.add('数据挖掘技能', dom3, is_label_show=True, label_pos='inside', is_legend_show=False)
treemap.render('拉勾网数据挖掘岗—技能图.html')



from scipy import stats
import pandas as pd
import psycopg2
# 获取数据库数据
conn = psycopg2.connect(database="lagou_job", user="postgres", password="774110919", host="127.0.0.1", port="5432")
cursor = conn.cursor()
sql = "select * from job"
df = pd.read_sql(sql, conn)
# 清洗数据,生成薪水列
dom = []
for i in df['job_salary']:
    i = ((float(i.split('-')[0].replace('k', '').replace('K', '')) + float(i.split('-')[1].replace('k', '').replace('K', ''))) / 2) * 1000
    dom.append(i)
df['salary'] = dom
# 去除无效列
data = df[df.job_experience != '不限']
# 生成不同工作经验的薪水列表
exp = []
for i in ['应届毕业生', '1-3年', '3-5年', '5-10年']:
    exp.append(data[data['job_experience'] == i]['salary'])
# 单因素方差分析
print(stats.f_oneway(*exp))



from pyecharts import Boxplot
import pandas as pd
import psycopg2

conn = psycopg2.connect(database="lagou_job", user="postgres", password="774110919", host="127.0.0.1", port="5432")
cursor = conn.cursor()
sql = "select * from job"
df = pd.read_sql(sql, conn)
dom22 = []
for i in df['job_experience']:
    if i in dom22:
        continue
    else:
        dom22.append(i)

dom = df[['job_experience', 'job_salary']]
data = [[], [], [], [], [], [], []]
dom1, dom2, dom3, dom4, dom5, dom6, dom7 = data
for i, j in zip(dom['job_experience'], dom['job_salary']):
    j = ((float(j.split('-')[0].replace('k', '').replace('K', '')) + float(j.split('-')[1].replace('k', '').replace('K', ''))) / 2) * 1000
    if i in ['不限']:
        dom1.append(j)
    elif i in ['应届毕业生']:
        dom2.append(j)
    elif i in ['1-3年']:
        dom4.append(j)
    elif i in ['3-5年']:
        dom5.append(j)
    else:
        dom6.append(j)

boxplot = Boxplot("拉勾网数据挖掘岗—工作经验薪水图(元/月)", title_pos='center', title_top='18', width=800, height=400)
x_axis = ['经验不限', '应届生', '1-3年', '3-5年', '5-10年']
y_axis = [dom1, dom2, dom4, dom5, dom6]
_yaxis = boxplot.prepare_data(y_axis)
boxplot.add("", x_axis, _yaxis)
boxplot.render("拉勾网数据挖掘岗—工作经验薪水图.html")



from scipy import stats
import pandas as pd
import psycopg2
# 获取数据库数据
conn = psycopg2.connect(database="lagou_job", user="postgres", password="774110919", host="127.0.0.1", port="5432")
cursor = conn.cursor()
sql = "select * from job"
df = pd.read_sql(sql, conn)
# 清洗数据,生成薪水列
dom = []
for i in df['job_salary']:
    i = ((float(i.split('-')[0].replace('k', '').replace('K', '')) + float(i.split('-')[1].replace('k', '').replace('K', ''))) / 2) * 1000
    dom.append(i)
df['salary'] = dom
# 去除无效列
data = df[df.job_education != '不限']
# 生成不同学历的薪水列表
edu = []
for i in ['大专', '本科', '硕士']:
    edu.append(data[data['job_education'] == i]['salary'])
# 单因素方差分析
print(stats.f_oneway(*edu))




import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd
import psycopg2
# 消除pandas输出省略号情况及换行情况
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# 获取数据库数据
conn = psycopg2.connect(database="lagou_job", user="postgres", password="774110919", host="127.0.0.1", port="5432")
cursor = conn.cursor()
sql = "select * from job"
df = pd.read_sql(sql, conn)
# 清洗数据,生成薪水列
dom = []
for i in df['job_salary']:
    i = ((float(i.split('-')[0].replace('k', '').replace('K', '')) + float(i.split('-')[1].replace('k', '').replace('K', ''))) / 2) * 1000
    dom.append(i)
df['salary'] = dom
# 去除学历不限
data = df[df.job_education != '不限']
# 去除工作经验不限及1年以下
data = data[data.job_experience != '不限']
data = data[data.job_experience != '1年以下']
# smf:最小二乘法,构建线性回归模型
anal = smf.ols('salary ~ C(job_experience) + C(job_education) + C(job_experience)*C(job_education)', data=data).fit()
# anova_lm:多因素方差分析
print(sm.stats.anova_lm(anal))
# 基本信息输出
print(anal.summary())




from pyecharts import Boxplot
import pandas as pd
import psycopg2

conn = psycopg2.connect(database="lagou_job", user="postgres", password="774110919", host="127.0.0.1", port="5432")
cursor = conn.cursor()
sql = "select * from job"
df = pd.read_sql(sql, conn)
dom22 = []
for i in df['job_education']:
    if i in dom22:
        continue
    else:
        dom22.append(i)

dom = df[['job_education', 'job_salary']]
data = [[], [], [], [], []]
dom1, dom2, dom3, dom4, dom5 = data
for i, j in zip(dom['job_education'], dom['job_salary']):
    j = ((float(j.split('-')[0].replace('k', '').replace('K', '')) + float(j.split('-')[1].replace('k', '').replace('K', ''))) / 2) * 1000
    if i in ['不限']:
        dom1.append(j)
    elif i in ['大专']:
        dom2.append(j)
    elif i in ['本科']:
        dom3.append(j)
    else:
        dom4.append(j)

boxplot = Boxplot("拉勾网数据挖掘岗—学历薪水图(元/月)", title_pos='center', title_top='18', width=800, height=400)
x_axis = ['学历不限', '大专', '本科', '硕士']
y_axis = [dom1, dom2, dom3, dom4]
_yaxis = boxplot.prepare_data(y_axis)
boxplot.add("", x_axis, _yaxis)
boxplot.render("拉勾网数据挖掘岗—学历薪水图.html")




import psycopg2
# 连接数据库
conn = psycopg2.connect(database="lagou_job", user="postgres", password="774110919", host="127.0.0.1", port="5432")
print("Opened database successfully")
# 创建工作表
cur = conn.cursor()
cur.execute('''CREATE TABLE job (id INT PRIMARY KEY NOT NULL, job_title TEXT NOT NULL, job_salary TEXT NOT NULL, job_city TEXT NOT NULL, job_experience TEXT NOT NULL, job_education TEXT NOT NULL, company_name TEXT NOT NULL, company_type TEXT NOT NULL, company_status TEXT NOT NULL, company_people TEXT NOT NULL, job_tips TEXT NOT NULL, job_welfare TEXT NOT NULL);''')
print("Table created successfully")
# 关闭数据库
conn.commit()
conn.close()



from pyecharts import Bar
import pandas as pd
import psycopg2

conn = psycopg2.connect(database="lagou_job", user="postgres", password="774110919", host="127.0.0.1", port="5432")
cursor = conn.cursor()
sql = "select * from job"
df = pd.read_sql(sql, conn)

place_message = df.groupby(['company_type'])
place_com = place_message['company_type'].agg(['count'])
place_com.reset_index(inplace=True)
place_com_last = place_com.sort_index()
dom = place_com_last.sort_values('count', ascending=False)[0:10]

attr = dom['company_type']
v1 = dom['count']
bar = Bar("拉勾网数据挖掘岗—公司类型TOP10", title_pos='center', title_top='18', width=800, height=400)
bar.add("", attr, v1, is_convert=True, xaxis_min=0, yaxis_rotate=30, yaxis_label_textsize=10, is_yaxis_boundarygap=True, yaxis_interval=0, is_label_show=True, is_legend_show=False, label_pos='right', is_yaxis_inverse=True, is_splitline_show=False)
bar.render("拉勾网数据挖掘岗—公司类型TOP10.html")