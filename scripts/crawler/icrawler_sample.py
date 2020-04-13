from icrawler.builtin import BaiduImageCrawler, BingImageCrawler, GoogleImageCrawler

bing_crawler = BingImageCrawler(downloader_threads=4,
                                storage={'root_dir': 'images'})
bing_crawler.crawl(keyword='cat', filters=None, offset=0, max_num=5)

# baidu_crawler = BaiduImageCrawler(storage={'root_dir': 'your_image_dir'})
# baidu_crawler.crawl(keyword='cat', offset=0, max_num=1000,
#                     min_size=(200,200), max_size=None)
