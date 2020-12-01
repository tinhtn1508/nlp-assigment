import requests
from bs4 import BeautifulSoup
from concurrent import futures
import multiprocessing
import os

CHILD_URL_FORMAT = 'https://vnexpress.net/bong-da/p{}'
MAX_WORKERS = multiprocessing.cpu_count() - 1
MAX_VNEXPRESS_PAGES = 2200
BATCH_SIZE = 100

def generate_vnexpress_urls():
    return [CHILD_URL_FORMAT.format(i) for i in range(1, MAX_VNEXPRESS_PAGES+1)]

def get_child_urls(root_url: str):
    child_container = []
    try:
        root_page = requests.get(root_url)
    except:
        print('Invalid url: {}'.format(root_url))
        return None
    soup = BeautifulSoup(root_page.content, 'html.parser')
    for element_h3 in soup.find_all("h2", {"class": "title-news"}):
        for element_a in element_h3.find_all("a", href=True):
            child_container.append(element_a.get("href"))
    return child_container

def worker(child_url: str):
    contents = []
    try:
        root_page = requests.get(child_url)
    except:
        print('Invalid url: {}'.format(root_url))
        return None
    soup = BeautifulSoup(root_page.content, 'html.parser')
    for article in soup.find_all('article', {"class": "fck_detail"}):
        for normal in soup.find_all('p', {"class": "Normal"}):
            contents.append(normal.text)
    return contents

def get_content_concurrency(child_urls):
    workers = min(MAX_WORKERS, len(child_urls))
    with futures.ThreadPoolExecutor(workers) as executor:
        res = executor.map(worker, child_urls)
    return list(res)

def save_to_file(name_file, data):
    path = '../data/' + name_file
    with open(path, 'a') as file:
        for line in data:
            if type(line) == str:
                file.writelines(line + '\n')
            else:
                print('value: {}, type: {}'.format(line, type(line)))

def get_child_url_concurrency(vnexpress_urls):
    workers = min(MAX_WORKERS, len(vnexpress_urls))
    with futures.ThreadPoolExecutor(workers) as executor:
        res = executor.map(get_child_urls, vnexpress_urls)
    total_url = []
    for ulrs in res:
        total_url.extend(ulrs)
    return total_url

if __name__ == "__main__":
    if not os.path.isfile('../data/vnexpress-urls-update.txt'):
        # vnexpress_urls = generate_vnexpress_urls()
        vnexpress_urls = ['https://vnexpress.net/bong-da']
        total_child_urls = get_child_url_concurrency(vnexpress_urls)
        print("Total child url: {}".format(len(total_child_urls)))
        save_to_file('vnexpress-urls.txt', total_child_urls)
    else:
        with open('../data/vnexpress-urls.txt', 'rb') as file:
            total_child_urls = file.readlines()
    
    number_batch = int(len(total_child_urls) / BATCH_SIZE) + 1
    for i in range(number_batch):
        start = i*BATCH_SIZE
        end = (i+1)*BATCH_SIZE if (i+1)*BATCH_SIZE < len(total_child_urls) else len(total_child_urls)
        contents = get_content_concurrency(total_child_urls[start: end])
        for content in contents:
            save_to_file('raw-vnexpress-contents-update.txt', content)