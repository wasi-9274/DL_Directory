# -*- coding: utf-8 -*-
from scrapy import Spider
from scrapy.http import Request, FormRequest
from time import sleep

class EplanSpider(Spider):
    name = 'eplan'
    rotate_user_agent = True
    allowed_domains = ['eplanning.ie']
    start_urls = ['http://eplanning.ie/']

    def parse(self, response):
        all_urls = response.xpath("//a/@href").extract()
        for url in all_urls:
            if "#" == url:
                pass
            else:
                yield Request(url, callback=self.parse_application)
                sleep(1)

    def parse_application(self, response):
        partial_url = response.xpath("//span[@class='glyphicon glyphicon-inbox btn-lg']/following-sibling::a/@href").extract_first()
        sleep(0.5)
        yield Request(response.urljoin(partial_url), callback=self.parse_form)

    def parse_form(self, response):
        yield FormRequest.from_response(response,
                                        dont_filter=True,
                                        formxpath='(//form)[2]',
                                        formdata={'RdoTimeLimit': '42'},
                                        callback=self.parse_pages)

    def parse_pages(self, response):
        application_urls = response.xpath("//td/a/@href").extract()
        for url in application_urls:
            sleep(0.8)
            yield Request(response.urljoin(url), callback=self.parse_items)
        next_page = response.xpath("//*[@rel='next']/@href").extract_first()
        sleep(0.9)
        yield Request(response.urljoin(next_page), callback=self.parse_pages)

    def parse_items(self, response):
        agent_btn = response.xpath('//*[@value="Agents"]/@style').extract_first()
        if 'display: inline;  visibility: visible;' in agent_btn:
            name = response.xpath('//tr[th="Name :"]/td/text()').extract_first()

            address_first = response.xpath('//tr[th="Address :"]/td/text()').extract()
            address_second = response.xpath('//tr[th="Address :"]/following-sibling::tr/td/text()').extract()[0:3]

            address = address_first + address_second

            phone = response.xpath('//tr[th="Phone :"]/td/text()').extract_first()

            fax = response.xpath('//tr[th="Fax :"]/td/text()').extract_first()

            email = response.xpath('//tr[th="e-mail :"]/td/a/text()').extract_first()

            url = response.url

            yield {'name': name,
                   'address': address,
                   'phone': phone,
                   'fax': fax,
                   'email': email,
                   'url': url}
        else:
            self.logger.info('Agent button not found on page, passing invalid url.')




