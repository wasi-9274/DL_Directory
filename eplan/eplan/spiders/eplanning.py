# -*- coding: utf-8 -*-
from scrapy import Spider
from scrapy.http import Request, FormRequest
from time import sleep
from random import randint


class EplanningSpider(Spider):
    name = 'eplanning'
    allowed_domains = ['eplanning.ie']
    start_urls = ['http://www.eplanning.ie/']

    def parse(self, response):
        urls = response.xpath("//a/@href").extract()
        for url in urls:
            if "#" == url:
                pass
            else:
                sleep(randint(1,2 ))
                yield Request(url, callback=self.parse_application)

    def parse_application(self, response):
        partial_url = response.xpath("//span[@class='glyphicon glyphicon-inbox btn-lg']/following-sibling::a/@href").extract_first()
        absolute_url = response.urljoin(partial_url)
        yield Request(absolute_url, callback=self.parse_form)

    def parse_form(self, response):
        yield FormRequest.from_response(response,
                                        dont_filter=True,
                                        formdata={'RdoTimeLimit': '42'},
                                        formxpath='(//form)[2]',
                                        callback=self.parse_pages)

    def parse_pages(self, response):
        application_urls = response.xpath("//td/a/@href").extract()
        for url in application_urls:
            sleep(randint(1, 2))
            yield Request(response.urljoin(url), self.parse_items)
        next_page_button = response.xpath('//li[@class="PagedList-skipToNext"]/a/@href').extract_first()
        yield Request(response.urljoin(next_page_button), callback=self.parse_pages)

    def parse_items(self, response):
        Agent_button = response.xpath('//input[@value="Agents"]/@style').extract_first()
        if "display: inline;  visibility: visible;" in Agent_button:
            Name =  response.xpath('//tr/th[text()="Name :"]/following-sibling::td/text()').extract_first()
            name = Name.strip()
            email = response.xpath("//td/a/text()").extract_first()

            yield {'Name': name,
                   'Email: ': email}

        else:
            self.logger.info("This particular application doesn't contain Agent button!")


