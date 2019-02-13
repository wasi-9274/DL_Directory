# -*- coding: utf-8 -*-
from scrapy import Spider
from selenium import webdriver
from time import sleep
from parsel import Selector
from selenium.webdriver.common.keys import Keys
from random import randint
import os
import requests
import shutil


class ImgurPicSpider(Spider):
    name = 'imgur_pic'
    allowed_domains = ['imgur.com']

    def start_requests(self):
        path = '/home/wasi/Desktop/vk_pics'
        driver = webdriver.Chrome('/home/wasi/Downloads/chromedriver')
        driver.get('https://imgur.com/')

        scroll_pause_time = randint(1, 3)
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            try:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                sel = Selector(text=driver.page_source)
                all_images = sel.xpath('//div[@class="Post-item-media"]/img/@src').extract()


                print("Sleeping for {} secs!!!!".format(scroll_pause_time))
                sleep(scroll_pause_time)
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
            except Exception as e:
                print(e)
        sleep(randint(1, 3))
        driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)

        sleep(randint(1, 3))
        driver.close()
