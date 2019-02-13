import os
import requests
import shutil
from scrapy.http import Request
from scrapy import Spider
from selenium import webdriver
from random import randint
from time import sleep
from parsel import Selector


class BookSpider(Spider):
    name = 'pics'
    allowed_domains = ['unsplash.com/t/animals']
    start_urls = ['https://unsplash.com/t/animals']

    def start_requests(self):
        global driver
        driver = webdriver.Chrome('/home/wasi/Downloads/chromedriver')
        sleep(randint(1, 2))
        url_main = 'https://unsplash.com/t/animals'
        driver.get(url_main)

        scroll_pause_time = 2
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            try:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                sel = Selector(text=driver.page_source)
                all_images_urls = sel.xpath('//a[@itemprop="contentUrl"]/@href').extract()
                for img_url in all_images_urls:
                    main_url = 'https://unsplash.com'+img_url
                    print('The image url is -> {}'.format(main_url))
                    yield Request(main_url, callback=self.download_images)
                self.logger.info("Sleeping for 5 secs!!!!")
                sleep(scroll_pause_time)
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    last_height = new_height
                    break
            except Exception as e:
                print(e)
                sleep(5)

    def download_images(self, response):
        print(response)
        driver.get(response.url)
        sleep(2)
        download_button = driver.find_element_by_xpath('//a[@title="Download photo"]')
        download_button.click()
        sleep(randint(2, 5))




