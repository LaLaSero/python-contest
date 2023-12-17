import os
import pandas as pd
import pickle

image_data = []
image_names = []

# 画像ファイルのディレクトリを指定
image_directory = os.path.expanduser('/Users/mypc/Desktop/mercari_images')

for filename in os.listdir(image_directory):
    if filename.endswith('.jpg'):
        # ファイル名から拡張子を除いた名前を抽出
        name = os.path.splitext(filename)[0]
        image_names.append(name)

        # 画像ファイルを開き、バイト配列に変換
        with open(os.path.join(image_directory, filename), 'rb') as file:
            img_byte_arr = file.read()
            image_data.append(img_byte_arr)

# DataFrameを作成
df = pd.DataFrame({'imageName': image_names, 'imageData': image_data})
print(df.info())

# pklファイルに保存
with open(os.path.expanduser('/Users/mypc/Desktop/mercari_images/mercari_dataset.pkl'), 'wb') as f:
    pickle.dump(df, f)