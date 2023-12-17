import pandas as pd
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import webbrowser
import io

# モデルのロード (VGG16を使用)
model = VGG16(weights='imagenet', include_top=False)

def extract_features(img_byte_arr, model):
    # バイト配列からPILイメージに変換
    img = Image.open(io.BytesIO(img_byte_arr))
    img = img.resize((32, 32))

    # PILイメージを配列に変換
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

def similarDetection():
	# pklファイルから画像データを読み込み
	df = pd.read_pickle("/Users/mypc/Desktop/mercari_images/mercari_dataset.pkl")

	# DataFrame内の画像の特徴ベクトルを抽出
	feature_list = [extract_features(img_data, model) for img_data in df['imageData']]

	# 比較対象の画像の特徴ベクトルを抽出
	target_img_path = "/Users/mypc/Desktop/image.jpg"
	target_img_features = extract_features(open(target_img_path, 'rb').read(), model)

	# 類似度の計算
	similarities = cosine_similarity([target_img_features], feature_list)[0]

	# 類似度スコアとインデックスのペアを作成し、スコアでソート
	similarity_scores = list(enumerate(similarities))
	similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

	# トップ3の類似画像を取得
	top_3_similar_images = similarity_scores[1:4]  # 最初の画像自体は除外

	# トップ3の類似画像のimageNameを表示
	for i, (index, score) in enumerate(top_3_similar_images):
		image_name = df['imageName'].iloc[index]
		print(f"Rank {i+1}, Image Name: {image_name}, Similarity Score: {score}")

		# MercariのURLを生成し、ブラウザで開く
		url = f'https://jp.mercari.com/item/{image_name}'
		print(f"Opening URL: {url}")
		webbrowser.open(url)
