import requests
import math
import time
import pymysql

db = pymysql.connect(host='localhost', user='root', password='你的数据库密码', port=3306, db='你的数据库名称')
cursor = db.cursor()
sql = 'CREATE TABLE IF NOT EXISTS lagou (' + \
      'id int(11) NOT NULL, ' + \
      'companyFullName VARCHAR(255)  NULL, ' + \
      'companyShortName VARCHAR(255) NULL, ' + \
      'city VARCHAR(255) NULL, ' + \
      'district VARCHAR(255) NULL, ' + \
      'companySize VARCHAR(255) NULL, ' + \
      'financeStage VARCHAR(255) NULL, ' + \
      'industryField VARCHAR(255) NULL, ' + \
      'companyLabelList VARCHAR(255) NULL, ' + \
      'positionName VARCHAR(255) NULL, ' + \
      'positionLables VARCHAR(255) NULL, ' + \
      'firstType VARCHAR(255) NULL, ' + \
      'salary VARCHAR(255) NULL, ' + \
      'workYear VARCHAR(255) NULL, ' + \
      'education VARCHAR(255) NULL, ' + \
      'jobNature VARCHAR(255) NULL, ' + \
      'positionAdvantage VARCHAR(255) NULL, ' + \
      'PRIMARY KEY (id))'
cursor.execute(sql)


def get_json(url, num):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36',
        'Host': 'www.lagou.com',
        'Referer': 'https://www.lagou.com/jobs/list_%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90?px=default&city=%E5%85%A8%E5%9B%BD',
        'X-Anit-Forge-Code': '0',
        'X-Anit-Forge-Token': 'None',
        'X-Requested-With': 'XMLHttpRequest'
    }

    data = {
        'first': 'true',
        'pn': num,
        'kd': '数据分析'}
    data = requests.post(url, headers=headers, data=data)
    data.raise_for_status()
    data.encoding = 'utf-8'
    # 得到包含职位信息的字典
    page = data.json()
    return page


def get_page_num(count):
    data = math.ceil(count / 15)
    # 这里选取的是100页，但拉勾网只显示30页
    if data > 100:
        return 100
    else:
        return data


def get_page_info(jobs_list, id):
    for job in jobs_list:
        id = id + 1
        companyFullName = job['companyFullName']
        companyShortName = job['companyShortName']
        city = job['city']
        district = job['district']
        companySize = job['companySize']
        financeStage = job['financeStage']
        industryField = job['industryField']
        companyLabelList = ','.join(job['companyLabelList'])
        positionName = job['positionName']
        positionLables = ','.join(job['positionLables'])
        firstType = job['firstType']
        salary = job['salary']
        workYear = job['workYear']
        education = job['education']
        jobNature = job['jobNature']
        positionAdvantage = job['positionAdvantage']
        cursor.execute(
            "INSERT INTO lagou(id,companyFullName,companyShortName,city,district,companySize,financeStage,industryField,companyLabelList,positionName,positionLables,firstType,salary,workYear,education,jobNature,positionAdvantage) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (str(id), companyFullName, companyShortName, city, district, companySize, financeStage, industryField,
             companyLabelList, positionName, positionLables, firstType, salary, workYear, education, jobNature,
             positionAdvantage))
        db.commit()
    return id


def main():
    url = 'https://www.lagou.com/jobs/positionAjax.json?px=default&needAddtionalResult=false'
    # 先设定页数为1,获取总的职位数
    page_1 = get_json(url, 1)
    total_count = page_1['content']['positionResult']['totalCount']
    num = get_page_num(total_count)
    time.sleep(20)
    print('职位总数:{},页数:{}'.format(total_count, num))
    m = 0
    for n in range(0, 101):
        # 对每个网页读取JSON, 获取每页数据
        page = get_json(url, n)
        jobs_list = page['content']['positionResult']['result']
        sum = get_page_info(jobs_list, m)
        m = m + 15
        print('已经抓取第{}页, 职位总数:{}'.format(n, sum))
        time.sleep(20)
    cursor.close()
    db.close()


if __name__ == "__main__":
    main()

import pandas as pd
import pymysql
import numpy as np

db = pymysql.connect(host='localhost', user='root', password='hexin123', port=3306, db='lagou')
cursor = db.cursor()
sql = "SELECT * FROM lagou"
data = pd.read_sql(sql, db)
db.close()

data.head(1)

import matplotlib.pyplot as plt  # 绘图
import matplotlib as mpl  # 配置字体
import seaborn as sns
from matplotlib.font_manager import FontProperties

myfont = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf', size=16)
sns.set(font=myfont.get_name())
mpl.rcParams['font.sans-serif'] = ['SimHei']
sns.set_style("whitegrid")
sns.set_context('talk')
# 城市分布
city = data['city'].value_counts()
plt.figure(figsize=(20, 20))
plt.pie(city.values, labels=city.index, autopct='%3.1f %%', textprops={'fontsize': 24, 'color': 'k'})
plt.show

# 公司规模
size = data['companySize'].value_counts()
plt.figure(figsize=(20, 10))
ax = sns.barplot(size.index, size.values, alpha=0.85)
ax.set_title("公司规模")
plt.show

# 公司融资情况
finance = data['financeStage'].value_counts()
plt.figure(figsize=(20, 10))
ax = sns.barplot(finance.index, finance.values, alpha=0.85)
ax.set_title("公司融资情况")
plt.show

# 公司的产业领域
# 按逗号区分
industry = []
for i in range(len(data)):
    if ',' in data['industryField'][i]:
        industry.extend(data['industryField'][i].split(','))
    elif '、' in data['industryField'][i]:
        industry.extend(data['industryField'][i].split('、'))
    else:
        industry.extend(data['industryField'][i].split(' '))
industry = pd.value_counts(industry)
from pyecharts import WordCloud

wordcloud = WordCloud(width=1000, height=600)
wordcloud.add("", industry.index, industry.values, word_size_range=[10, 100], is_more_utils=True)
wordcloud.render("industryField.html")

# 公司福利
label = []
for i in range(len(data)):
    label.extend(data['companyLabelList'][i].split(','))
label = pd.value_counts(label)
wordcloud = WordCloud(width=1000, height=600)
wordcloud.add("", label.index, label.values, word_size_range=[10, 100], is_more_utils=True)
wordcloud.render("companyLabelList.html")

# 职位标签
positionlabel = []
for i in range(len(data)):
    positionlabel.extend(data['positionLables'][i].split(','))
positionlabel = pd.value_counts(positionlabel)
wordcloud = WordCloud(width=1000, height=600)
wordcloud.add("", positionlabel.index, positionlabel.values, word_size_range=[10, 100], is_more_utils=True)
wordcloud.render("positionLables.html")

# 职位所在部门
type = data['firstType'].value_counts()
plt.figure(figsize=(20, 20))
plt.pie(type.values, labels=type.index, autopct='%3.1f %%', textprops={'fontsize': 24})
plt.title("所在一级部门")
plt.show

# 薪水
salary = data['salary'].value_counts()
plt.figure(figsize=(20, 10))
ax = sns.barplot(x=salary.index[:20], y=salary.values[:20])
ax.set_xticklabels(salary.index, rotation=90)

# 将薪水下限和上限分开，并取其均值
data['salary_low'] = data['salary'].apply(
    lambda x: int(x.split('-')[0].split('k')[0]) if 'k' in x else int(x.split('-')[0].split('K')[0]))
data['salary_up'] = data['salary'].apply(
    lambda x: int(x.split('-')[1].split('k')[0]) if 'k' in x else int(x.split('-')[1].split('K')[0]))
data['salary_mean'] = 1
for i in range(len(data)):
    a = data['salary_low'][i]
    b = data['salary_up'][i]
    data.loc[i, 'salary_mean'] = (a + b) / 2

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
sns.distplot(data['salary_low'], ax=axes[0], color='r', kde=True)
sns.distplot(data['salary_up'], ax=axes[1], color='y', kde=True)
plt.show()

plt.figure(figsize=(10, 10))
plt.scatter(x=data['salary_low'], y=data['salary_up'], color='m')
plt.xlabel('薪水下限')
plt.ylabel('薪水上限')
plt.xlim(0, 70)
plt.show()

plt.figure(figsize=(15, 10))
ax = sns.distplot(data['salary_mean'], hist=True, color="g", kde_kws={"shade": True})
plt.show()

# 要求的工作经验
workyear = data['workYear'].value_counts()
plt.figure(figsize=(20, 10))
sns.barplot(x=workyear.index, y=workyear.values)
ax.set_title('工作经验要求')
plt.show()

# 学历要求
education = data['education'].value_counts()
plt.figure(figsize=(10, 10))
ax = sns.barplot(x=education.index, y=education.values, alpha=0.5)
ax.set_title('学历要求')
plt.show()

# 工作性质
jobnature = data['jobNature'].value_counts()
plt.figure(figsize=(10, 10))
plt.pie(jobnature.values, labels=jobnature.index, autopct='%3.1f %%', textprops={'fontsize': 20, 'color': 'r'})
plt.title('学历要求')
plt.show()

# 双变量分析
# 城市与薪水
plt.figure(figsize=(20, 10))
sns.boxplot(data["city"], data["salary_mean"])
sns.stripplot(data["city"], data["salary_mean"])
plt.title('城市与薪水', fontsize=20)
plt.show()

# 公司规模与薪水
plt.figure(figsize=(15, 10))
sns.pointplot(data["companySize"], data["salary_mean"], markers='*', linestyles="--", color='m', capsize=0.3)
sns.swarmplot(data["companySize"], data["salary_mean"])
plt.title('公司规模与薪水的关系', fontsize=20)
plt.show()

# 融资情况与薪水
plt.figure(figsize=(15, 10))
sns.pointplot(data["financeStage"], data["salary_mean"], markers='^', linestyles="-", color='g', capsize=0.1)
plt.title('融资情况与薪水的关系', fontsize=20)
plt.show()

# 工作经验与薪水
plt.figure(figsize=(15, 10))
sns.boxplot(data["workYear"], data["salary_mean"])
plt.title('工作经验与薪水的关系', fontsize=20)
plt.show()

# 学历与薪水
plt.figure(figsize=(10, 10))
sns.boxplot(data["education"], data["salary_mean"])
sns.stripplot(x="education", y="salary_mean", data=data, color='y', alpha=".25")
plt.title('学历与薪水的关系', fontsize=20)
plt.show()

# 为分析产业与其他变量的关系，我们取排第一的作为其产业
data['filed'] = data['industryField'].apply(
    lambda x: x.split(',')[0] if ',' in x else x.split('、')[0] if '、' in x else x.split(' ')[0])
# 产业与薪水
plt.figure(figsize=(20, 10))
sns.boxplot(data["filed"], data["salary_mean"])
sns.stripplot(x="filed", y="salary_mean", data=data, color='y', alpha=".25")
plt.title('产业与薪水的关系', fontsize=20)
plt.show()

# 三变量分析
# 工作经验、学历与薪水
plt.figure(figsize=(15, 10))
sns.pointplot(data["workYear"], data["salary_mean"], hue=data['education'])
plt.title('工作经验、学历与薪水的关系', fontsize=20)
plt.show()

# 公司规模、学历
plt.figure(figsize=(15, 10))
sns.countplot(data["companySize"], hue=data['education'])
plt.show()