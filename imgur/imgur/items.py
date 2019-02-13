# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class ImgurItem(scrapy.Item):
    images_url = scrapy.Field()
    images = scrapy.Field()
