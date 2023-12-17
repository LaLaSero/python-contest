import os
import pandas as pd
import pickle

def generate_zozoPkl():
	image_data = []
	categories = []

	# 画像ファイルのディレクトリを指定
	image_directory = os.path.expanduser('./zozo_images')

	for filename in os.listdir(image_directory):
		if filename.endswith('.jpg'):
			# カテゴリ名を抽出
			category = filename.split('_')[0]
			categories.append(category)

			# 画像ファイルを開き、バイト配列に変換
			with open(os.path.join(image_directory, filename), 'rb') as file:
				img_byte_arr = file.read()
				image_data.append(img_byte_arr)

	# DataFrameを作成
	df = pd.DataFrame({'fashionImage': image_data, 'fashionCategory': categories})
	print(df.info())

	# pklファイルに保存
	with open(os.path.expanduser('/Users/mypc/Desktop/zozo_images/zozo_dataset.pkl'), 'wb') as f:
		pickle.dump(df, f)