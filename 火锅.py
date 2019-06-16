import time
import requests
from pyquery import PyQuery as pq
import pandas as pd

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36'}

def restaurant(url):
    # 获取网页静态源代码
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.text
    except Exception:
        return None

name=[]
url = []
star = []
comment = []
avg_price = []
taste = []
environment = []
services = []
recommend = []

num = {'hs-OEEp': 0, 'hs-4Enz': 2, 'hs-GOYR': 3, 'hs-61V1': 4, 'hs-SzzZ': 5, 'hs-VYVW': 6, 'hs-tQlR': 7, 'hs-LNui': 8, 'hs-42CK': 9}
def detail_number(htm):
    try:
        a = str(htm)
        a = a.replace('1<', '<span class="1"/><')
        a = a.replace('.', '<span class="."/>')
        b = pq(a)
        cn = b('span').items()
        number = ''
        for i in cn:
            attr = i.attr('class')
            if attr in num:
                attr = num[attr]
            number = number + str(attr)
        number = number.replace('None', '')
    except:
        number = ''
    return number

def info_restaurant(html):
    # 获取饭店的名称和链接
    doc = pq(html)
    for i in range(1,16):
        #获取饭店名称
        shop_name = doc('#shop-all-list > ul > li:nth-child('+str(i)+') > div.txt > div.tit > a:nth-child(1) > h4').text()
        if shop_name == '':
            break
        name.append(shop_name)
        #获取饭店链接
        url.append(doc('#shop-all-list > ul > li:nth-child('+str(i)+') > div.pic > a').attr('href'))
        try:
            star.append(doc('#shop-all-list > ul > li:nth-child('+str(i)+') > div.txt > div.comment > span').attr('title'))
        except:
            star.append("")
        #获取评论数量
        comment_html = doc('#shop-all-list > ul > li:nth-child('+str(i)+') > div.txt > div.comment > a.review-num > b')
        comment.append(detail_number(comment_html))
        #获取人均消费
        avg_price_html = doc('#shop-all-list > ul > li:nth-child('+str(i)+') > div.txt > div.comment > a.mean-price > b')
        avg_price.append(detail_number(avg_price_html))
        #获取口味评分
        taste_html = doc('#shop-all-list > ul > li:nth-child('+str(i)+') > div.txt > span > span:nth-child(1) > b')
        taste.append(detail_number(taste_html))
        #获取环境评分
        environment_html = doc('#shop-all-list > ul > li:nth-child('+str(i)+') > div.txt > span > span:nth-child(2) > b')
        environment.append(detail_number(environment_html))
        #获取服务评分
        services_html = doc('#shop-all-list > ul > li:nth-child('+str(i)+') > div.txt > span > span:nth-child(3) > b')
        services.append(detail_number(services_html))
        #推荐菜,都是显示三道菜
        try:
            recommend.append(doc('#shop-all-list > ul > li:nth-child('+str(i)+') > div.txt > div.recommend > a:nth-child(2)').text()+str(',')+\
                            doc('#shop-all-list > ul > li:nth-child('+str(i)+') > div.txt > div.recommend > a:nth-child(3)').text()+str(',')+\
                            doc('#shop-all-list > ul > li:nth-child('+str(i)+') > div.txt > div.recommend > a:nth-child(4)').text())
        except:
            recommend.append("")
for i in range(1,51):
    print('正在获取第{}页饭店信息'.format(i))
    hotpot_url = 'http://www.dianping.com/chengdu/ch10/g110p'+str(i)+'?aid=93195650%2C68215270%2C22353218%2C98432390%2C107724883&cpt=93195650%2C68215270%2C22353218%2C98432390%2C107724883&tc=3'
    html = restaurant(hotpot_url)
    info_restaurant(html)
    print ('第{}页获取成功'.format(i))
    time.sleep(12)

shop = {'name': name, 'url': url, 'star': star, 'comment': comment, 'avg_price': avg_price, 'taste': taste, 'environment': environment, 'services': services, 'recommend': recommend}
shop = pd.DataFrame(shop, columns=['name', 'url', 'star', 'comment', 'avg_price','taste', 'environment', 'services', 'recommend'])
shop.to_csv("shop.csv",encoding="utf_8_sig",index = False)



import pandas as pd
import matplotlib.pyplot as plt #绘图
import matplotlib as mpl #配置字体
import seaborn as sns

data = pd.read_csv('shop.csv')
data = data.dropna(axis=0,how='any')
print(data.head(2))
len(data)

from matplotlib.font_manager import FontProperties
myfont=FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf',size=16)
sns.set(font=myfont.get_name())
mpl.rcParams['font.sans-serif'] = ['SimHei']
sns.set_style("whitegrid")
sns.set_context('talk')

#星级情况
plt.figure(figsize=(15,10))
sns.countplot(data["star"],order = ['五星商户','准五星商户','四星商户','准四星商户','三星商户'])
plt.show()

#评论数情况
plt.figure(figsize=(15,10))
sns.distplot(data["comment"],kde=True,rug=True,color='g')
plt.show()

#星级和评论的关系
plt.figure(figsize=(15,10))
sns.barplot(data["star"],y=data["comment"],order = ['五星商户','准五星商户','四星商户','准四星商户','三星商户'])
plt.show()

#平均价格
plt.figure(figsize=(15,10))
sns.distplot(data["avg_price"],kde=True,rug=True,color='m')
plt.show()

#价格和星级
plt.figure(figsize=(15,10))
sns.violinplot(data["star"],y=data["avg_price"],palette="Set2",order = ['五星商户','准五星商户','四星商户','准四星商户','三星商户'])
plt.show()

#价格和评论数量
plt.figure(figsize=(15,15))
sns.jointplot(data["avg_price"], data["comment"], kind="hex")
plt.show()

#口味，环境，服务得分分布情况
fig, axes = plt.subplots(3,1,figsize=(10, 30))
sns.distplot(data['taste'], ax = axes[0],color = 'r', kde = True)
sns.distplot(data['environment'], ax = axes[1], color = 'y', kde=True)
sns.distplot(data['services'], ax = axes[2], color = 'g', kde=True)
plt.show()


#口味，环境，服务得分与星级的关系
plt.figure(figsize=(10,10))
sns.boxplot(data["star"],data["taste"],order = ['五星商户','准五星商户','四星商户','准四星商户','三星商户'],palette="Set3")
sns.stripplot(x="star", y="taste", data=data, color='y',alpha=".25",order = ['五星商户','准五星商户','四星商户','准四星商户','三星商户'])
plt.show()

plt.figure(figsize=(10,10))
sns.boxplot(data["star"],data["environment"],order = ['五星商户','准五星商户','四星商户','准四星商户','三星商户'],palette="Set3")
sns.stripplot(x="star", y="environment", data=data, color='r',alpha=".25",order = ['五星商户','准五星商户','四星商户','准四星商户','三星商户'])
plt.show()

plt.figure(figsize=(10,10))
sns.boxplot(data["star"],data["services"],order = ['五星商户','准五星商户','四星商户','准四星商户','三星商户'],palette="Set3")
sns.stripplot(x="star", y="services", data=data, alpha=".25",order = ['五星商户','准五星商户','四星商户','准四星商户','三星商户'])
plt.show()

#口味，环境，服务得分与评论数量，平均价格的关系
plt.figure(figsize=(18,18))
df = data[['star','comment', 'avg_price','taste', 'environment', 'services']]
sns.pairplot(df,hue='star',palette="husl")
plt.show()

#口味，环境，服务
plt.figure(figsize=(15,15))
df = data[['star','taste', 'environment', 'services']]
sns.pairplot(df,hue='star',palette="Set2",kind="reg")
plt.show()

#特色菜
import jieba
delicious = []
for i in range(750):
    try:
        recommend = jieba.lcut(data['recommend'][i])
        while ',' in recommend:
            recommend.remove(',')
        while '（' in recommend:
            recommend.remove('（')
        while '）' in recommend:
            recommend.remove('）')
        delicious.extend(recommend)
    except:
        continue
delicious = pd.value_counts(delicious)
from pyecharts import WordCloud
wordcloud = WordCloud(width=1000, height=600)
wordcloud.add("", delicious.index, delicious.values, word_size_range=[12, 150], is_more_utils=True)
wordcloud.render("delicious.html")

#
from sklearn.cluster import KMeans
#为更好聚类，我们将星级转为数字
for i in range(750):
    try:
        if data.loc[i,'star'] == '五星商户':
             data.loc[i,'star_score'] = 5
        elif data.loc[i,'star'] == '准五星商户':
            data.loc[i,'star_score'] = 4.5
        elif data.loc[i,'star'] == '四星商户':
            data.loc[i,'star_score'] = 4
        elif data.loc[i,'star'] == '准四星商户':
            data.loc[i,'star_score'] = 3.5
        else:
            data.loc[i,'star_score'] = 3
    except:
        continue

estimator = KMeans(n_clusters=3)#构造聚类器
data_1 = data[['star_score','taste', 'environment', 'services']]
estimator.fit(data_1)#聚类
label_pred = estimator.labels_ #获取聚类标签
centroids = estimator.cluster_centers_ #获取聚类中心
inertia = estimator.inertia_ # 获取聚类准则的总和

#绘制k-means结果
x0 = data_1[label_pred == 0]
x1 = data_1[label_pred == 1]
x2 = data_1[label_pred == 2]
plt.figure(figsize=(8,8))
plt.scatter(x0['taste'], x0['environment'], c = "red", marker='o', label='非常推荐')
plt.scatter(x1['taste'], x1['environment'], c = "green", marker='*', label='不推荐')
plt.scatter(x2['taste'], x2['environment'], c = "blue", marker='+', label='一般推荐')
plt.xlabel('taste')
plt.ylabel('environment')
plt.legend()
plt.title('口味与环境')
plt.show()

plt.figure(figsize=(8,8))
plt.scatter(x0['taste'], x0['services'], c = "red", marker='o', label='一般推荐')
plt.scatter(x1['taste'], x1['services'], c = "green", marker='*', label='不推荐')
plt.scatter(x2['taste'], x2['services'], c = "blue", marker='+', label='非常推荐')
plt.xlabel('taste')
plt.ylabel('services')
plt.legend()
plt.title('口味与服务')
plt.show()

plt.figure(figsize=(8,8))
plt.scatter(x0['environment'], x0['services'], c = "red", marker='o', label='一般推荐')
plt.scatter(x1['environment'], x1['services'], c = "green", marker='*', label='不推荐')
plt.scatter(x2['environment'], x2['services'], c = "blue", marker='+', label='非常推荐')
plt.xlabel('environment')
plt.ylabel('services')
plt.legend()
plt.title('环境与服务')
plt.show()

plt.figure(figsize=(8,8))
plt.scatter(x0['star_score'], x0['services'], c = "red", marker='o', label='一般推荐')
plt.scatter(x1['star_score'], x1['services'], c = "green", marker='*', label='不推荐')
plt.scatter(x2['star_score'], x2['services'], c = "blue", marker='+', label='非常推荐')
plt.xlabel('star_score')
plt.ylabel('services')
plt.legend()
plt.title('星级与服务')
plt.show()

data['label'] = label_pred
tuijian = data[data['label'] == 2]

#推荐其中平均价格偏低，并且销量也偏低的店铺
comment_low = int(input('评论数量下限：'))
comment_high = int(input('评论数量上限：'))
avg_price_low = int(input('人均消费下限：'))
avg_price_hign = int(input('人均消费上限：'))
special_dish = input('想吃的特色菜：')

tj = tuijian[(tuijian['comment']>comment_low)&(tuijian['comment']<comment_high) & (tuijian['avg_price']>avg_price_low)& (tuijian['avg_price']<avg_price_hign)].reset_index()
for i in range(len(tj)):
    if special_dish in tj['recommend'][i]:
        print(tj['name'][i])