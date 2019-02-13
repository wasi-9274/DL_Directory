# -*- coding: utf-8 -*-
from scrapy import Spider
from selenium import webdriver
from parsel import Selector
from scrapy.http import Request
from time import sleep
from selenium.webdriver.common.keys import Keys


class MapshowSpider(Spider):
    name = 'mapshow'
    allowed_domains = ['npe18.mapyourshow.com']

    def start_requests(self):
        self.driver = webdriver.Chrome("/home/wasi/Downloads/chromedriver")
        self.driver.get("https://npe18.mapyourshow.com/7_0/alphalist.cfm?endrow=64&alpha=*")

        scroll_pause_time = 2
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            try:
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                self.logger.info("Sleeping for 5 secs!!!!")
                sleep(scroll_pause_time)
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
            except Exception as e:
                print(e)
        sleep(5)
        self.driver.find_element_by_tag_name('body').send_keys(Keys.CONTROL + Keys.HOME)
        sel = Selector(text=self.driver.page_source)
        partial_urls = sel.xpath("//td[@class='mys-table-exhname']//a/@href").extract()
        print(partial_urls)
        for url in partial_urls:
            try:
                main_url = "https://npe18.mapyourshow.com" + url
                yield Request(main_url, callback=self.parse_page)
            except Exception as e:
                print(e)

    def parse_page(self, response):
        pass





