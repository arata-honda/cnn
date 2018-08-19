import numpy as np
import cv2
import time
import random

class normalized_dataset:
    def __init__(self):
        self.path_and_labels = []
        self.data_set = []

    def load_txt(self):
        # パスとラベルが記載されたファイルから、それらのリストを作成する
        print('loading path and label txt...')
        start = time.time()
        with open('path_and_label.txt', mode='r', encoding='utf8') as file:
            for line in file:
                # 改行を除く
                line = line.rstrip()
                # スペースで区切られたlineを、リストにする
                line_list = line.split()
                self.path_and_labels.append(line_list)
        elapsed_time = time.time() - start
        # 上から順にとってきているのでランダムにする。(トレーニング・テスト分割用)
        random.shuffle(self.path_and_labels)
        print(f"done! elapsed_time:{elapsed_time}[sec]")

    def normalized_resize_image(self, image_path):
        # 画像を読み込み、サイズを変更し、0〜1の値に正規化する
        img = cv2.imread(image_path)
        img = cv2.resize(img, (28, 28))

        return img.flatten().astype(np.float32) / 255.0

    def make_data_set(self):
        self.load_txt()
        print('making dataset...')
        start = time.time()
        for path_and_label in self.path_and_labels:
            tmp_list = []

            normalized_img = self.normalized_resize_image(image_path=path_and_label[0])
            tmp_list.append(normalized_img)

            # 分類するクラス数の長さを持つ仮のN次元配列を作成する
            classes_array = np.zeros(5, dtype='float64')
            # ラベルの数字によって、リストを更新する
            classes_array[int(path_and_label[1])] = 1
            tmp_list.append(classes_array)

            self.data_set.append(tmp_list)

        elapsed_time = time.time() - start
        print(f"done! elapsed_time:{elapsed_time}[sec]")


    def get_data_set(self):
        return self.data_set

if __name__ == '__main__':
    normalized_dataset = normalized_dataset()
    normalized_dataset.make_data_set()
    data_set = normalized_dataset.get_data_set()
    print(data_set[0][0].shape)
    print(data_set[1][0])
