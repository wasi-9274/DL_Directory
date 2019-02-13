# -*- coding: utf-8 -*-
import scrapy
from scrapy import Request


class CraigSpider(scrapy.Spider):
    name = 'craig'
    allowed_domains = ['craigslist.org']
    start_urls = ['http://newyork.craigslist.org/search/egr/']

    def parse(self, response):
        all_urls = response.xpath('//time[@class="result-date"]/following-sibling::a/@href').extract()
        for url in all_urls:
            yield Request(url, callback=self.parse_items)

        yield Request(response.urljoin(response.xpath('//a[@class="button next"]/@href').extract_first()),
                      callback=self.parse)

    def parse_items(self, response):
        p = []
        title = response.xpath('//span[@id="titletextonly"]/text()').extract_first()
        paragraph = response.xpath('//section[@id="postingbody"]/text()').extract()

        yield {'Title': title,
               'paragraph': paragraph}


