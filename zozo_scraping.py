from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import requests
import os

def zozo_scraping():
	# Create the 'images' directory on the Desktop
	image_dir = os.path.expanduser('/Users/mypc/Desktop/zozo_images')
	os.makedirs(image_dir, exist_ok=True)

	category_list = ['sneakers', 'slip-on', 'sandal', 'pumps', 'boots', 'dress-shoes', 'loafers', 'rain-shoes']
	for category in category_list:
		count = 0  # カウンターをカテゴリごとにリセット
		for page_number in range(1, 6):  # pno=1からpno=5まで
			search_query = category
			url = f"https://zozo.jp/category/shoes/{search_query}/?pno={page_number}"
			target_class = 'p-search-list'

			options = webdriver.ChromeOptions()
			service = webdriver.chrome.service.Service(ChromeDriverManager().install())
			browser = webdriver.Chrome(service=service, options=options)
			browser.get(url)

			try:
				height = browser.execute_script("return document.body.scrollHeight")
				while height > 0:
					browser.execute_script(f"window.scrollTo(0, {height});")
					height -= 512
					time.sleep(0.5)

				image_elements = WebDriverWait(browser, 5).until(
					EC.presence_of_all_elements_located((By.CSS_SELECTOR, f".{target_class} img"))
				)
				
				for img in image_elements:
					if count >= 1000:  # カウンターの上限を設定
						break
					img_url = img.get_attribute('src')
					if img_url:
						response = requests.get(img_url)
						image_path = os.path.join(image_dir, f'{search_query}_{count}.jpg')
						with open(image_path, 'wb') as file:
							file.write(response.content)
					count += 1
			except Exception as e:
				print(f"An error occurred: {e}")
			finally:
				browser.quit()

			print(f"Completed downloading images for {search_query} page {page_number}.")