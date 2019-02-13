# -*- coding: utf-8 -*-
from scrapy import Spider
from scrapy import Request
from random import randint
from time import sleep


class QuoteSpider(Spider):
    name = 'quote'
    allowed_domains = ['brainyquote.com']
    start_urls = ['https://www.brainyquote.com/topics']

    def parse(self, response):
        all_urls = response.xpath('//a[@class="topicIndexChicklet"]/@href').extract()
        for url in all_urls:
            sleep(randint(1, 2))
            yield Request(response.urljoin(url), callback=self.parse_items)

    def parse_items(self, response):
        class_quotes = response.xpath("//div[@class='clearfix']")
        for quote in class_quotes:
            sleep(randint(1,3))
            quote_text = quote.xpath(".//a[@title='view quote']/text()").extract_first()
            author_name = quote.xpath('.//a[@title="view author"]/text()').extract_first()
            yield{"Quote": quote_text,
                  "Author": author_name}



