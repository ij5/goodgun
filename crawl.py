from icrawler.builtin import BingImageCrawler

classes = ['cats', 'trees', 'roads', 'Human faces']

number = 20

for c in classes:
    bing_crawler = BingImageCrawler(storage={'root_dir': f'datasets/p/{c.replace(" ", "_")}'})
    bing_crawler.crawl(keyword=c, filters=None, max_num=number, offset=0)

