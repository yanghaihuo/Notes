# --*--coding:utf-8 --*--
import requests
import json
# from multiprocessing import Pool
# from multiprocessing.dummy import Pool
import random
import time
from selenium import webdriver
import hashlib
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")


os.chdir(r'C:\Users\Lenovo\Desktop\taobao')
def taobao(keyword,num_pages=20):
    url = 'https://h5.m.taobao.com/'

    # 获取所需cookies
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    driver = webdriver.Chrome(chrome_options=chrome_options)
    driver.maximize_window()
    driver.get(url)
    time.sleep(2)
    cookies_dic = {}
    cookies_list = driver.get_cookies()
    for i in cookies_list:
        cookies_dic[i['name']] = i['value']
        if i['name'] == '_m_h5_tk':
            token = i['value'].split("_")[0]
    driver.quit()
    a=[]
    for page in range(1,num_pages + 1):
        # 计算Sign
        t = int(time.time() * 1000)
        data = '{"event_submit_do_new_search_auction":"1","_input_charset":"utf-8","topSearch":"1","atype":"b","searchfrom":"1","action":"home:redirect_app_action","from":"1","q":"' + \
               keyword + '","sst":"1","n":20,"buying":"buyitnow","m":"api4h5","token4h5":"","abtest":"16","wlsort":"16","style":"list","closeModues":"nav,selecthot,onesearch","page":' + str(
            page) + '}'
        src = token + '&' + str(t) + '&12574478&' + data
        m2 = hashlib.md5()
        m2.update(src.encode("utf8"))
        sign = m2.hexdigest()

        url = 'https://acs.m.taobao.com/h5/mtop.taobao.wsearch.h5search/1.0/'
        params = {'jsv': '2.3.16',
                  'appKey': '12574478',
                  't': t,
                  'sign': sign,
                  'api': 'mtop.taobao.wsearch.h5search',
                  'v': '1.0',
                  'H5Request': 'true',
                  'ecode': '1',
                  'type': 'jsonp',
                  'dataType': 'jsonp',
                  'callback': 'mtopjsonp2',
                  'data': data}

        headers = {
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Host': 'acs.m.taobao.com',
            'Pragma': 'no-cache',
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 11_0 like Mac OS X) AppleWebKit/604.1.38 (KHTML, like Gecko) Version/11.0 Mobile/15A372 Safari/604.1'}

        resp = requests.get(url, headers=headers, params=params, cookies=cookies_dic,verify=False,)
        resp_json = json.loads(resp.text.replace('mtopjsonp2(','')[:-1])
        goodsList = resp_json['data']['listItem']
        print(page)
        for goods in goodsList:
            userId=goods['userId']
            goodsurl = "https://item.taobao.com/item.htm?id=" + goods['item_id']
            title=goods['title']
            nick=goods['nick']
            location=goods['location']
            originalPrice=goods['originalPrice']
            priceWap=goods['priceWap']
            try:
                commentCount = goods['commentCount']
            except:
                commentCount=' '
            shipping = goods['shipping']
            sold = goods['sold']
            try:
                zkType = goods['zkType']
            except:
                zkType=' '
            img2=goods['img2']
            m=[userId,goodsurl,title,nick,location,originalPrice,priceWap,commentCount,shipping,sold,zkType,img2]
            print(m)
            a.append(m)
            time.sleep(random.uniform(1, 2))
    test = pd.DataFrame(a)
    test.columns=['userId','goodsurl','title','nick','location','originalPrice','priceWap','commentCount','shipping','sold','zkType','img2']
    test.to_csv('淘宝' + keyword + '.csv', encoding='utf_8_sig',index=False)






if __name__ == '__main__':
	keyword = input('请输入关键词:')
	print()
	num_pages = int(input('请输入爬取的页数:'))
	print()
	taobao(keyword, num_pages)