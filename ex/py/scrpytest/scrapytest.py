import scrapy							
import urllib.request							
							
class Book4Spider(scrapy.Spider):							
    name='book_image_link'							
    allowed_domains=['wikibook.co.kr']							
    start_urls=['https://wikibook.co.kr/list/']							
							
							
    def parse(self, response):							
        li_list=response.css('.col-md-1')							
        for i in li_list[:5]:							
            url=i.css("::attr(src)").extract_first()							
            # print(url)							
            li=url.split('/')							
            file_name=li[-1]							
            urllib.request.urlretrieve(url,'C:/shjung/'+file_name)							
        yield print(url, file_name)							
            #response.follow(response.urljoin(href), self.parse_book)							