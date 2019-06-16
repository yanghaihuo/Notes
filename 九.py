# # 按城市等级合并
# first_city = [u'北京', u'上海', u'广州', u'深圳']
# new_first_city = [u'成都', u'杭州', u'武汉', u'重庆', u'南京', u'天津', u'苏州', u'西安', u'长沙', u'沈阳', u'青岛', u'郑州', u'大连', u'东莞',
#                   u'宁波']
# second_city = [u'厦门', u'福州', u'无锡', u'合肥', u'昆明', u'哈尔滨', u'佛上', u'长春', u'温州', u'石家庄', u'南宁', u'常州', u'泉州', u'南昌',
#                u'贵阳', u'太原', u'烟台', u'嘉兴', u'南通', u'金华', u'珠海', u'惠州', u'徐州', u'海口', u'乌鲁木齐', u'绍兴', u'中山', u'台州',
#                u'兰州']
# third_city = [u'潍坊', u'保定', u'镇江', u'扬州', u'桂林', u'唐山', u'三亚', u'湖州', u'呼和浩特', u'廊坊', u'洛阳', u'威海', u'盐城', u'临沂', u'江门',
#               u'汕头', u'泰州', u'漳州', u'邯郸', u'济宁', u'芜湖', u'淄博', u'银川', u'柳州', u'绵阳', u'湛江', u'鞍山', u'赣州', u'大庆', u'宜昌',
#               u'包头', u'咸阳', u'秦皇岛', u'株洲', u'莆田', u'吉林', u'淮安', u'肇庆', u'宁德', u'衡阳', u'南平', u'连云港', u'丹东', u'丽江', u'揭阳',
#               u'延边朝鲜族自治州', u'舟山', u'九江', u'龙岩', u'沧州', u'抚顺', u'襄阳', u'上饶', u'营口', u'三明', u'蚌埠', u'丽水', u'岳阳', u'清远',
#               u'荆州', u'泰安', u'衢州', u'盘锦', u'东营', u'南阳', u'马鞍山', u'南充', u'西宁', u'孝感', u'齐齐哈尔']
#
# def citycombine(s):
#     if s in first_city:
#         return u'新一线'
#     if s in new_first_city:
#         return u'一线'
#     if s in second_city:
#         return u'二线'
#     if s in third_city:
#         return u'三线'
#     if s == u'0':
#         return u'未知'
#     if s == u'不详':
#         return u'未知'
#     else:
#         return u'其他'
#
# train_master['UserInfo_2'] = train_master['UserInfo_2'].apply(lambda x: citycombine(x))
# len(train_master['UserInfo_2'].unique())
#
#
# def province_encode(s):
#     if s == u'不详':
#         return '0'
#     if s == u'广东':
#         return '1'
#     if s == u'山东':
#         return '2'
#     if s == u'江苏':
#         return '3'
#     if s == u'浙江':
#         return '4'
#     if s == u'四川':
#         return '5'
#     if s == u'福建':
#         return '6'
#     if s == u'湖南':
#         return '7'
#     else:
#         return '8'
#
# train_master['UserInfo_7'] = train_master['UserInfo_7'].apply(lambda x: province_encode(x))
# len(train_master['UserInfo_7'].unique())
#
# # 抽样
# # 按个数
# print(df.sample(n=5))
# # 按百分比，150 * 0.001 * 100 取15个
# print(df.sample(frac=0.001))
# print(df['satisfaction_level'].sample(5))
# # 出现的比例
# np_s.value_counts(normalize=True)
#
# iris_data['class'] = iris_data['class'].apply(lambda x: x.split('-')[1])
#
# from sklearn.datasets import load_iris
# iris=load_iris()
# attributes=iris.data  #获取属性数据
# target=iris.target  #获取类别数据，这里注意的是已经经过了处理，target里0、1、2分别代表三种类别
# labels=iris.feature_names#获取类别名字
# iris_data .columns=['sepal_lengh_cm','sepal_width_cm','petal_length_cm','petal_width_cm','species']


# -*- coding: utf-8 -*-
# 仅供交流学习
import requests
from pyquery import PyQuery as pq
import time, random
import os
os.chdir(r'C:\Users\Lenovo\Desktop')
# 请求头 cookie 必须需要加上,爬前request网址试下可以get到全内容不,不能的话换下cookie
headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
           'Accept-Encoding': 'gzip, deflate,br',
           'Accept-Language': 'zh-CN,zh;q=0.9',
           'Cache-Control': 'no-cache',
           'Connection': 'keep-alive',
           'Cookie': 'uuid=02724092-c319-4ba3-89e5-7bf432065f79; ganji_uuid=1367370650065045912399; Hm_lvt_936a6d5df3f3d309bda39e92da3dd52f=1528970077,1528970098; cityDomain=sz; lg=1; clueSourceCode=%2A%2300; sessionid=7273bfb7-b191-4334-f256-30ec9febf860; cainfo=%7B%22ca_s%22%3A%22sem_baiduss%22%2C%22ca_n%22%3A%22bdpc_sye%22%2C%22ca_i%22%3A%22-%22%2C%22ca_medium%22%3A%22-%22%2C%22ca_term%22%3A%22%25E4%25BA%258C%25E6%2589%258B%25E8%25BD%25A6%22%2C%22ca_content%22%3A%22-%22%2C%22ca_campaign%22%3A%22-%22%2C%22ca_kw%22%3A%22-%22%2C%22keyword%22%3A%22-%22%2C%22ca_keywordid%22%3A%2249888355177%22%2C%22scode%22%3A%2210103188612%22%2C%22ca_transid%22%3Anull%2C%22platform%22%3A%221%22%2C%22version%22%3A1%2C%22client_ab%22%3A%22-%22%2C%22guid%22%3A%2202724092-c319-4ba3-89e5-7bf432065f79%22%2C%22sessionid%22%3A%227273bfb7-b191-4334-f256-30ec9febf860%22%7D; antipas=2496334kL545T396w008C21ez41; preTime=%7B%22last%22%3A1531722065%2C%22this%22%3A1528850701%2C%22pre%22%3A1528850701%7D',
           'Host': 'www.guazi.com',
           'Pragma': 'no-cache',
           'Referer': 'https://www.guazi.com/sz/buy/o1r3_16_6/',
           'Upgrade-Insecure-Requests': '1',
           'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'}
# headers ={
#     'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36',
#     # cookie 必须需要加上,爬前request网址试下可以get到全内容不,不能的话换下cookie
#     'Cookie':'uuid=2276c0ec-9654-440b-b583-788b4a4b7788; ganji_uuid=1526893422532532003974; lg=1; Hm_lvt_936a6d5df3f3d309bda39e92da3dd52f=1531222100; clueSourceCode=10103000312%2300; antipas=2496334IBPB545K39a6U008T21x41; sessionid=94cd5b73-23cf-4515-d5a1-b7b03461d083; cainfo=%7B%22ca_s%22%3A%22pz_baidu%22%2C%22ca_n%22%3A%22tbmkbturl%22%2C%22ca_i%22%3A%22-%22%2C%22ca_medium%22%3A%22-%22%2C%22ca_term%22%3A%22-%22%2C%22ca_content%22%3A%22-%22%2C%22ca_campaign%22%3A%22-%22%2C%22ca_kw%22%3A%22-%22%2C%22keyword%22%3A%22-%22%2C%22ca_keywordid%22%3A%22-%22%2C%22scode%22%3A%2210103000312%22%2C%22ca_transid%22%3Anull%2C%22platform%22%3A%221%22%2C%22version%22%3A1%2C%22client_ab%22%3A%22-%22%2C%22guid%22%3A%222276c0ec-9654-440b-b583-788b4a4b7788%22%2C%22sessionid%22%3A%2294cd5b73-23cf-4515-d5a1-b7b03461d083%22%7D; close_finance_popup=2018-07-16; cityDomain=sz; preTime=%7B%22last%22%3A1531725774%2C%22this%22%3A1531222098%2C%22pre%22%3A1531222098%7D'
# }
# 代理ip
proxies = {
    'http': 'http://60.177.231.103:18118',
    'https': 'http://60.177.231.103:18118'
}


# proxies = {
#     'http':'http://60.177.226.225:18118',
#     'https':'http://60.177.226.225:18118'
# }
# proxies = {
#     'http':'http://14.118.252.228:6666',
#     'https':'http://14.118.252.228:6666'
# }


class GuaziSpider():

    # 初始化爬虫
    def __init__(self):
        # 目标url
        self.baseurl = 'https://www.guazi.com'
        '''
        在进行接口测试的时候，我们会调用多个接口发出多个请求，在这些请求中有时候需要保持一些共用的数据，例如cookies信息。
        requests库的session对象能够帮我们跨请求保持某些参数，也会在同一个session实例发出的所有请求之间保持cookies。
        '''
        self.s = requests.Session()
        self.s.headers = headers
        # 本地ip被封的话启用该处ip代理池
        # self.s.proxies = proxies
        # 其中www代表瓜子二手车全国车源,如果只想爬某个城市的,如深圳的用sz替换www
        self.start_url = 'https://www.guazi.com/www/buy/'
        self.infonumber = 0  # 用来记录爬取了多少条信息用

    # get_page用来获取url页面
    def get_page(self, url):
        return pq(self.s.get(url).text)

    # page_url用来生成第n到第m页的翻页链接
    def page_url(self, n, m):
        page_start = n
        page_end = m
        # 新建空列表用来存翻页链接
        page_url_list = []
        for i in range(page_start, page_end + 1, 1):
            base_url = 'https://www.guazi.com/www/buy/o{}/#bread'.format(i)
            page_url_list.append(base_url)

        return page_url_list

    # detail_url用来抓取详情页链接
    def detail_url(self, start_url):
        # 获取star_url页面
        content = self.get_page(start_url)
        # 解析页面,获取详情页链接content=pq(self.s.get(start_url).text)
        for chref in content('ul[@class="carlist clearfix js-top"]  > li > a').items():
            url = chref.attr.href
            detail_url = self.baseurl + url
            yield detail_url

    # carinfo用来抓取每辆车的所需信息
    def carinfo(self, detail_url):
        content = self.get_page(detail_url)
        d = {}
        d['model'] = content('h2.titlebox').text().strip()  # 车型
        d['registertime'] = content('ul[@class="assort clearfix"] li[@class="one"] span').text()  # 上牌时间
        d['mileage'] = content('ul[@class="assort clearfix"] li[@class="two"] span').text()  # 表显里程
        d['secprice'] = content('span[@class="pricestype"]').text()  # 报价
        d['newprice'] = content('span[@class="newcarprice"]').text()  # 新车指导价(含税)
        d['address'] = content('ul[@class="assort clearfix"]').find('li'). \
            eq(2).find('span').text()  # 上牌地
        d['displacement'] = content('ul[@class="assort clearfix"]'). \
            find('li').eq(3).find('span').text()  # 排量
        return d

    def run(self, n, m):
        page_start = n
        page_end = m
        with open('guazidata{}to{}.txt'.format(page_start, page_end), 'a', encoding='utf-8') as f:
            for pageurl in self.page_url(page_start, page_end):
                print(pageurl)
                print("让俺歇歇10-15秒啦，太快会关小黑屋的！")
                time.sleep(random.randint(10, 15))
                for detail_url in self.detail_url(pageurl):
                    print(detail_url)
                    d = self.carinfo(detail_url)
                    f.write(d['model'] + ',')
                    f.write(d['registertime'] + ',')
                    f.write(d['mileage'] + ',')
                    f.write(d['secprice'] + ',')
                    f.write(d['newprice'] + ',')
                    f.write(d['address'] + ',')
                    f.write(d['displacement'] + '\n')
                    time.sleep(0.3)
                    self.infonumber += 1
                    print('爬了%d辆车,continue!' % self.infonumber)
                print('+' * 10)


if __name__ == '__main__':
    gzcrawler = GuaziSpider()
    # 这儿改数字,例如:爬取第1页到第100页的信息
    gzcrawler.run(1, 100)
