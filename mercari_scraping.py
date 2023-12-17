from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import requests
import os
from selenium.common.exceptions import TimeoutException
import re

def extract_id_with_prefix_from_url(url):
	pattern = r"/(m\d+)_"
	match = re.search(pattern, url)
	return match.group(1) if match else ""

def find_index_of_matching_element(category_list, query):
	try:
		return category_list.index(query)
	except ValueError:
		return -1

def mercari_scraping(response):
	category_list = ['スニーカー', 'スリッパ', 'サンダル', 'パンプス', 'ブーツ', 'ドレスシューズ',
					'ローファー', 'モカシン', '長靴']
	category_list_eng = ['sneakers', 'slip-on', 'sandal', 'pumps', 'boots',
					'dress-shoes', 'loafers', 'moccasins', 'rain-shoes']
	search_query = category_list[find_index_of_matching_element(category_list_eng, response)]
	url = f"https://jp.mercari.com/search?keyword={search_query}"
	target_class = 'sc-bcd1c877-2 gWHcGv'

	options = webdriver.ChromeOptions()
	service = webdriver.chrome.service.Service(ChromeDriverManager().install())
	browser = webdriver.Chrome(service=service, options=options)
	browser.get(url)
	time.sleep(6)
	# make directory to save images
	image_dir = "{}_images".format(category_list_eng[k])
	os.makedirs(image_dir, exist_ok=True)

	count = 0
	try:
		# get image elements

		height = browser.execute_script("return document.body.scrollHeight")
		while height > 0:
			browser.execute_script(f"window.scrollTo(0, {height});")
			height -= 512
			time.sleep(0.5)
		try:
			image_elements = WebDriverWait(browser, 5).until(
			EC.presence_of_all_elements_located((By.CSS_SELECTOR, f".sc-bcd1c877-2.gWHcGv img"))
			)
			image_urls = [element.get_attribute('src') for element in image_elements]
		except TimeoutException:
			image_elements = None
		print(image_urls)
		
		for i, img in enumerate(image_elements):
			if count >= 115:
				break
			# get url of image from img element
			img_url = img.get_attribute('src')
			print(img_url)
			if img_url:
				# download image and save
				response = requests.get(img_url)
				image_path = os.path.join(image_dir, f'{extract_id_with_prefix_from_url(img_url)}.jpg')
				with open(image_path, 'wb') as file:
					file.write(response.content)
			count += 1
	except Exception as e:
		print(f"an error occured: {e}")
	finally:
		browser.quit()
		print("complete downloading images.")


