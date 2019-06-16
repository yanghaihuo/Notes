import itchat
import matplotlib.pyplot as plt
from collections import Counter
import re
import jieba
from wordcloud import WordCloud, ImageColorGenerator
from scipy.misc import imread
import pandas as pd
import os
import jieba.analyse
import numpy as np
from PIL import Image
from snownlp import SnowNLP
import csv

print(os.getcwd()) #获取当前工作路径
os.chdir(r'C:\Users\Lenovo\Desktop')
# 可视化的中文处理
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')

# 登陆微信
itchat.auto_login(hotReload=True)

# 获取自己的好友列表
friends = itchat.get_friends(update=True)
# print(friends)

# 好友总数, 第一个是自己
total = len(friends[1:])
print ('好友总数:', total)

# 查看好友男友比例
male = female = other = 0
for friend in friends[1:]:
    sex = friend['Sex']
    if sex == 1:
        male += 1
    elif sex == 2:
        female += 1
    else:
        other += 1
    # print(male)
print(male)
print(female)
print(other)
sexes = list()
sexes.append(float(male) * 100 / total)
sexes.append(float(female) * 100 / total)
sexes.append(float(other) * 100 / total)

print ('男性好友: %-.2f%%' % sexes[0])
print ('女性好友: %-.2f%%' % sexes[1])
print ('不明身份好友: %-.2f%%' % sexes[2])


all_information=[]
NickNames=[]
RemarkNames=[]
Sexs=[]
Provinces=[]
Citys=[]
Signatures=[]
for friend in friends:
    NickNames.append(friend['NickName'])
    RemarkNames.append(friend['RemarkName'])
    Sexs.append(friend['Sex'])
    Provinces.append(friend['Province'])
    Citys.append(friend['City'])
    Signatures.append(friend['Signature'])
    all_information=[NickNames,RemarkNames,Sexs,Provinces,Citys,Signatures]
test = pd.DataFrame(all_information).T
# test.rename(columns={0:'NickNames',1:'Sexs',2:'Provinces',3:'Citys',4:'Signatures'}, inplace=True)
test.columns=['NickName','RemarkName','Sex','Province','City','Signature']
# print(test.info())
test.to_csv('weixin.csv', encoding='utf_8_sig',index=False)

# DataFrame.to_csv(path_or_buf=None, sep=', ', na_rep='', float_format=None, columns=None,
# header=True,index=True, index_label=None, mode='w', encoding=None, compression=None,
# quoting=None, quotechar='"',line_terminator='\n', chunksize=None, tupleize_cols=None,
#  date_format=None, doublequote=True,escapechar=None, decimal='.')



signatures = list()
provinces = list()
cities = list()
# 过滤掉签名中的冗余字符
pattern = re.compile('<[^>]+>')
for friend in friends:
    if friend['Signature']:
        signature = pattern.sub('', friend['Signature']).strip().replace('&amp;', '')
        signatures.append(signature)
    provinces.append(friend['Province'] if friend['City'] else 'NULL')
    cities.append(friend['City'] if friend['City'] else 'NULL')


# 绘制好友省份柱状图
pro = sorted(Counter(provinces).most_common(10), key=lambda d: -d[1])
label, num = zip(*pro)
#draw(num, label, 'Provinces', 'nums', 'Province distribute of my friends')

# 绘制好友城市柱状图
cit = sorted(Counter(cities).most_common(10), key=lambda d: -d[1])
label, num = zip(*cit)
#draw(num, label, 'Cities', 'nums', 'City distribute of my friends')







def analyseSignature(friends):
    signatures = ''
    emotions = []
    pattern = re.compile("1f\d.+")
    for friend in friends:
        signature = friend['Signature']
        if(signature != None):
            signature = signature.strip().replace('span', '').replace('class', '').replace('emoji', '')
            signature = re.sub(pattern,'',signature)
            if(len(signature)>0):
                nlp = SnowNLP(signature)
                # print(nlp)
                # print(nlp.sentences)
                # print(nlp.words) # 分词
                # print(nlp.sentiments) # 计算正向概率 权值
                # print(nlp.tags) # 对词的属性进行分类
                # print(nlp.pinyin) #汉转拼音
                # print(nlp.tf)  # 词频
                # print(nlp.idf)  # 逆向文件频率
                emotions.append(nlp.sentiments)
                signatures += ' '.join(jieba.analyse.extract_tags(signature,5))
    with open('signatures.txt','wt',encoding='utf-8') as file:
         file.write(signatures)

    # Sinature WordCloud
    # back_coloring = np.array(Image.open('heart.jpg'))
    wordcloud = WordCloud(
        font_path='simfang.ttf',
        background_color="white",
        max_words=1200,
        # mask=back_coloring,
        max_font_size=75,
        random_state=45,
        width=960,
        height=720,
        margin=15
    )

    wordcloud.generate(signatures)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    wordcloud.to_file('signatures.jpg')

    # Signature Emotional Judgment
    count_good = len(list(filter(lambda x:x>0.66,emotions)))
    # print(list(filter(lambda x:x>0.66,emotions)))
    count_normal = len(list(filter(lambda x:x>=0.33 and x<=0.66,emotions)))
    count_bad = len(list(filter(lambda x:x<0.33,emotions)))
    labels = [u'负面消极',u'中性',u'正面积极']
    values = (count_bad,count_normal,count_good)
    plt.xlabel(u'情感判断')
    plt.ylabel(u'频数')
    plt.xticks(range(3),labels)
    plt.legend(loc='upper right',)
    plt.bar(range(3), values, color = 'rgb')
    plt.title('杨海的微信好友签名信息情感分析')
    plt.show()


def analyseSex(firends):
    sexs = list(map(lambda x:x['Sex'],friends[1:]))
    counts = list(map(lambda x:x[1],Counter(sexs).items()))
    labels = ['Unknow','Male','Female']
    colors = ['red','yellowgreen','lightskyblue']
    plt.figure(figsize=(8,5), dpi=80)
    plt.axes(aspect=1)
    plt.pie(counts, #性别统计结果
            labels=labels, #性别展示标签
            colors=colors, #饼图区域配色
            labeldistance = 1.1, #标签距离圆点距离
            autopct = '%3.1f%%', #饼图区域文本格式
            shadow = False, #饼图是否显示阴影
            startangle = 90, #饼图起始角度
            pctdistance = 0.6 #饼图区域文本距离圆点距离
    )
    plt.legend(loc='upper right',)
    plt.title(u'%s' % friends[0]['NickName'])
    plt.show()


def analyseLocation(friends):
    headers = ['NickName','RemarkName','Sex','Province','City','Signature']
    with open('weixin1.csv','w',encoding="utf_8_sig",newline='') as csvFile:
        writer = csv.DictWriter(csvFile, headers)
        writer.writeheader()
        for friend in friends[1:]:
           row = {}
           row['NickName'] = friend['NickName']
           row['RemarkName'] = friend['RemarkName']
           row['Sex'] = friend['Sex']
           row['Province'] = friend['Province']
           row['City'] = friend['City']
           row['Signature'] = friend['Signature']
           print(row)
           writer.writerow(row)

# 登陆微信
itchat.auto_login(hotReload=True)
# 获取自己的好友列表
friends = itchat.get_friends(update=True)
# 好友总数, 第一个是自己

total = len(friends[1:])
print ('好友总数:', total)

analyseSignature(friends)
# analyseSex(friends)
# analyseLocation(friends)