from scrapy import Spider
from scrapy.http import Request


class SpringSpider(Spider):
    name = 'spring'
    allowed_domains = ['raleighspringshow.com']
    start_urls = ['https://raleighspringshow.com/exhibitor-list/exhibitors/']

    def parse(self, response):
        urls = response.xpath("//h2/a/@href").extract()
        for url in urls:
            yield Request(response.urljoin(url), callback=self.parse_items)

    def parse_items(self, response):
        Company_name = response.xpath("//h2[@class='listing-title']/text()").extract_first()
        Company_name = Company_name.replace('\n', '').strip()
        booth = response.xpath("//div[@class='listing-boothNum']/text()").extract_first()
        booth_update = booth.replace('Booth: ', '').strip()
        address = response.xpath('//div[@class="listing-address-line-1"]/text()').extract_first()
        city_zip = response.xpath("//div[@class='citystatezip']/text()").extract_first()
        city_zip_update = city_zip.strip()
        categories = response.xpath('//li/span/text()').extract_first()
        categories_update = categories.strip()



        yield {'Company_name': Company_name,
                'booth': booth_update,
                'address': address,
                'city_zip': city_zip_update,
                'categories': categories}
