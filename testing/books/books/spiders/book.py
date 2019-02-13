# -*- coding: utf-8 -*-
from scrapy import Spider
from scrapy.http import requests
from selenium import webdriver
from random
from time import sleep


class BookSpider(Spider):
    name = 'book'
    allowed_domains = ['unsplash.com/t/animals']
    start_urls = ['https://unsplash.com/t/animals']

    def start_requests(self, response):
        driver = webdriver.Chrome('/home/wasi/Downloads/chromedriver')
        sleep(randint(1,2))
        url_main = driver.get('https://unsplash.com/t/animals')
        sel = Selector(text=driver.page_source)




        sleep(randint(4,7))
        driver.close()
