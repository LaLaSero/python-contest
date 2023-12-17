import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def classify_image(model_path, image_path, category_list):
    # モデルのロード
    model = load_model(model_path)

    # 画像の読み込みと前処理
    img = image.load_img(image_path, target_size=(32, 32))  # モデルに合わせたサイズに調整
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # 正規化

    # 予測の実行
    predictions = model.predict(img_array)
    predicted_category = category_list[np.argmax(predictions)]

    # 結果の表示と返却
    print(f"The image is classified as: {predicted_category}")
    return predicted_category


def categoryDetection():
	# モデルファイルのパス
	model_path = '/Users/mypc/Documents/myproject/python-automation-program-contest/my_model.keras'

	# 画像ファイルのパス
	image_path = '/Users/mypc/Desktop/image.jpg'

	# カテゴリリスト
	category_list = ['sneakers', 'slip-on', 'sandal', 'pumps', 'boots', 'dress-shoes', 'loafers', 'rain-shoes']

	# 関数の呼び出し
	category = classify_image(model_path, image_path, category_list)
