from scrapy import Spider
from selenium import webdriver
from time import sleep
from selenium.common.exceptions import NoSuchElementException
from parsel import Selector
from scrapy.http import Request

class BooksSpider(Spider):
    name = 'books'
    allowed_domains = ['bookstoscrape.com']
    start_urls = ['http://books.toscrape.com/']

    def start_requests(self):
        self.driver = webdriver.Chrome("/home/wasi/Downloads/chromedriver")
        self.driver.get("http://books.toscrape.com/")
        sel = Selector(text=self.driver.page_source)
        books_url = sel.xpath("//h3/a/@href").extract()
        for url in books_url:
            main_url = "http://books.toscrape.com/" + url
            yield Request(main_url, callback=self.parse_books)

        while True:
            try:
                next_button = self.driver.find_element_by_xpath("//a[contains(text(), 'next')]")
                next_button.click()
                self.logger.info("Sleeping page for 3 seconds")
                sleep(3)

                sel = Selector(text=self.driver.page_source)
                books_url = sel.xpath("//h3/a/@href").extract()
                for url in books_url:
                    main_url = "http://books.toscrape.com/catalogue/" + url
                    yield Request(main_url, callback=self.parse_books)

            except NoSuchElementException:
                self.logger.info("Element not found!")
                self.driver.quit()
                break

    def parse_books(self, response):
        book_img = response.xpath("//div[@class='item active']/img/@src").extract_first()
        book_img = book_img.replace("../..", "http://books.toscrape.com")

        book = response.xpath("//h1/text()").extract_first()

        price = response.xpath("//p[@class='price_color']/text()").extract_first()

        upc = response.xpath('//tr[th="UPC"]/td/text()').extract_first()

        product_type = response.xpath('//tr[th="Product Type"]/td/text()').extract_first()

        available = response.xpath('//tr[th="Availability"]/td/text()').extract_first()

        yield {'book_img_url': book_img,
               'book_name': book,
               'price': price,
               'upc': upc,
               'product_type': product_type,
               'availibility': available}

















