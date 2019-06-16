# import pandas as pd
# from pyecharts import Pie,Line,Scatter
# import os
# import numpy as np
# import jieba
# import jieba.analyse
# from wordcloud import WordCloud,ImageColorGenerator
# import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties
# import datetime
#
#
#
# font = FontProperties(fname=r'c:\windows\fonts\simsun.ttc')#,size=20指定本机的汉字字体位置
# os.chdir(r'C:\Users\Lenovo\Desktop')
# datas = pd.read_csv('doupo.csv',index_col = 0,encoding = 'gbk')
#
# # texts = ''.join(datas.content.values)
# texts=";".join('%s' %id for id in datas.content.tolist())
# cut_text = " ".join(jieba.cut(texts))
# # TF_IDF
# keywords = jieba.analyse.extract_tags(cut_text, topK=500, withWeight=True, allowPOS=('a','e','n','nr','ns'))
# text_cloud = dict(keywords)
# pd.DataFrame(keywords).to_excel('TF_IDF关键词前500.xlsx')
#
#
#
# # bg = plt.imread("abc.jpg")
# # 生成
# wc = WordCloud(# FFFAE3
#     background_color="white",  # 设置背景为白色，默认为黑色
#     width=1600,  # 设置图片的宽度
#     height=1200,  # 设置图片的高度
#     # mask=bg,
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
# # bg_color = ImageColorGenerator(bg)
# # plt.imshow(wc.recolor(color_func=bg_color))
# # plt.imshow(wc)
# # 为云图去掉坐标轴
# plt.axis("off")
# plt.show()
# wc.to_file("斗破苍穹评论词云.png")


# coding:utf-8
import csv
import time
from wxpy import *

# 将要发送的好友的名字存到list中
FRIENDS = ['xx', 'xxxx']
CSV_PATH = './文件名'


# 定义函数获取csv中的内容
def read_csv(path):
    f = open(path, 'r', encoding='utf8')
    reader = csv.DictReader(f)
    # for info in reader:
    #     print(info)
    return [info for info in reader]


# 定义获取发送内容的函数
def get_msg(infos, name):
    template = "{name}，提醒下，{time}记得来参加{event}，地点在{location}，{note}"
    for info in infos:
        if info['微信昵称'] == name:
            msg = template.format(
                name=info['微信昵称'],
                time=info['时间'],
                event=info['事件'],
                location=info['地址'],
                note=info['备注']
            )
            return msg
    # 如果在infos列表中没有找到对应的微信昵称，则输出None
    return None


# 定义用于群发操作的函数
def send_to_friends(infos, friends):
    # 初始化微信机器人
    bot = Bot()
    for friend in friends:
        # 搜素好友
        try:  # 新增的异常处理
            friend_search = bot.friends().search(friend)
        except ResponseError as e:
            print(e.err_code, e.err_msg)
        # 如果搜索结果仅有一个，则发送，否则返回错误信息
        if (len(friend_search) == 1):
            msg = get_msg(infos, friend)
            print(msg)
            if msg:
                try:  # 新增的异常处理
                    friend_search[0].send(msg)
                except ResponseError as e:
                    print(e.err_code, e.err_msg)
            else:
                print("发送失败！用户名不在csv中：" + friend)
        else:
            print("发送失败！请检查用户名：" + friend)
        time.sleep(3)


# 调用群发函数
send_to_friends(read_csv(CSV_PATH), FRIENDS)