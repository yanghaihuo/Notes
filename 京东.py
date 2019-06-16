from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium import webdriver
from bs4 import BeautifulSoup
# import pymongo
import time

# 连接数据库
# client = pymongo.MongoClient(host='localhost', port=27017)
# db = client.JD_products
# collection = db.products

# 启动浏览器
browser = webdriver.Chrome()
wait = WebDriverWait(browser, 50)


# def to_mongodb(data):
#     # 存储数据信息
#     try:
#         collection.insert(data)
#         print("Insert The Data Successfully")
#     except:
#         print('Insert The Data Failed')


def search():
    browser.get('https://www.jd.com/')
    try:
        # 查找搜索框及搜索按钮，输入信息并点击按钮
        input = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#key")))
        submit = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#search > div > div.form > button")))
        input[0].send_keys('笔记本')
        submit.click()
        # 查找笔记本按钮及销量按钮，依次点击按钮
        button_1 = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#J_selector > div:nth-child(2) > div > div.sl-value > div.sl-v-list > ul > li:nth-child(1) > a")))
        button_1.click()
        button_2 = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#J_filter > div.f-line.top > div.f-sort > a:nth-child(2)")))
        button_2.click()
        # 获取总页数
        page = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '#J_bottomPage > span.p-skip > em:nth-child(1) > b')))
        return page[0].text
    except TimeoutException:
        search()


def next_page(page_number):
    try:
        # 滑动到网页底部，加载出所有商品信息
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(10)
        html = browser.page_source
        parse_html(html)
        # 当网页到达100页时，下一页按钮失效，所以选择结束程序
        while page_number == 101:
            exit()
        # 查找下一页按钮，并点击按钮
        button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '#J_bottomPage > span.p-num > a.pn-next > em')))
        button.click()
        wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#J_goodsList > ul > li:nth-child(60)")))
        # 判断翻页成功
        wait.until(EC.text_to_be_present_in_element((By.CSS_SELECTOR, "#J_bottomPage > span.p-num > a.curr"), str(page_number)))
    except TimeoutException:
        return next_page(page_number)


def parse_html(html):
    """
    解析商品列表网页
    """
    data = {}
    soup = BeautifulSoup(html, 'html.parser')
    goods_info = soup.select('.gl-item')
    # 查看当前页商品数量，了解是否还有未加载的商品
    quantity = 'item: ' + str(len(goods_info))
    print(quantity)
    for info in goods_info:
        # 获取商品标题信息
        title = info.select('.p-name.p-name-type-2 a em')[0].text.strip()
        title = title.replace('爱心东东', '')
        print("title: ", title)
        data['_id'] = title
        # 获取商品价格信息
        price = info.select('.p-price i')[0].text.strip()
        price = int(float(price))
        print("price: ", price)
        data['price'] = price
        # 获取商品的评论数量
        commit = info.select('.p-commit strong')[0].text.strip()
        commit = commit.replace('条评价', '')
        if '万' in commit:
            commit = commit.split("万")
            commit = int(float(commit[0]) * 10000)
        else:
            commit = int(float(commit.replace('+', '')))
        print("commit: ", commit)
        data['commit'] = commit
        # 获取商品的商店名称
        shop_name = info.select('.p-shop a')
        if (len(shop_name)) == 1:
            print("shop_name: ", shop_name[0].text.strip())
            data['shop_name'] = shop_name[0].text.strip()
        else:
            print("shop_name: ", '京东')
            data['shop_name'] = '京东'
        # 获取商品的商店属性
        shop_property = info.select('.p-icons i')
        if (len(shop_property)) >= 1:
            message = shop_property[0].text.strip()
            if message == '自营':
                print("shop_property: ", message)
                data['shop_property'] = message
            else:
                print("shop_property: ", '非自营')
                data['shop_property'] = '非自营'
        else:
            print("shop_property: ", '非自营')
            data['shop_property'] = '非自营'
        # to_mongodb(data)
        print(data)
        print("\n\n")


def main():
    total = int(search())
    print(total)
    for i in range(2, total+2):
        time.sleep(20)
        print("第", i-1, "页：")
        next_page(i)


if __name__ == "__main__":
    main()




# from selenium import webdriver
# from selenium.common.exceptions import TimeoutException
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.support.wait import WebDriverWait
# from pyquery import PyQuery as pq
# from urllib.parse import quote
# import time
# import csv
#
#
# # chrome_options = webdriver.ChromeOptions()
# chrome_options.add_argument('--headless')
# browser = webdriver.Chrome(chrome_options=chrome_options)
# wait = WebDriverWait(browser, 10)
#
# def index_page(page):
#     print('正在爬取第', page, '页')
#     try:
#         url = 'https://s.taobao.com/search?q=' + quote('手机')
#         browser.get(url)
#         if page > 1:
#             input = wait.until(
#                 EC.presence_of_element_located((By.CSS_SELECTOR, '#mainsrp-pager div.form > input')))
#             submit = wait.until(
#                 EC.element_to_be_clickable((By.CSS_SELECTOR, '#mainsrp-pager div.form > span.btn.J_Submit')))
#             input.clear()
#             input.send_keys(page)
#             submit.click()
#         wait.until(
#             EC.text_to_be_present_in_element((By.CSS_SELECTOR, '#mainsrp-pager li.item.active > span'), str(page)))
#         wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.m-itemlist .items .item')))
#         get_products()
#     except TimeoutException:
#         index_page(page)
#
# def get_products():
#     html = browser.page_source
#     doc = pq(html)
#     items = doc('#mainsrp-itemlist .items .item').items()
#     for item in items:
#         a=item.find('.title').text()
#         b=item.find('.price').text()
#         c=item.find('.deal-cnt').text()
#         d=item.find('.shop').text()
#         e=item.find('.location').text()
#         f=item.find('.pic .img').attr('data-src')
#         product = {
#             '名称': a,
#             '价格': b,
#             '销量': c,
#             '店铺': d,
#             '地址': e,
#             '图片': f
#         }
#         print(product)
#         with open('淘宝女包前20页数据.csv', 'a', encoding='utf-8',newline='') as csvfile:
#             fieldnames = ['名称', '价格', '销量', '店铺', '地址', '图片']
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writeheader()
#             writer.writerow({'名称': a, '价格': b, '销量': c, '店铺': d, '地址': e, '图片': f})
#             time.sleep(1)
#
# def main():
#     MAX_PAGE=10
#     for i in range(1, MAX_PAGE + 1):
#         index_page(i)
#     browser.close()
#
# if __name__ == '__main__':
#     main()