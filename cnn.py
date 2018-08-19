import numpy as np
import random
import tensorflow as tf
import os
import normalized_dataset as nd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 識別ラベルの数
NUM_CLASSES = 5
# 学習する時の画像のサイズ(px)
IMAGE_SIZE = 28
# 画像チャネル(カラーなので3)
IMAGE_CHANNEL = 3
# 画像の次元数(28 * 28 * カラー)
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNEL
# 2x2の畳み込みを二回かけているので全結合層では画像の縦横はそれぞれ1/4になる
FULL_CONNECTIVE_LAYER_SIZE = int(IMAGE_SIZE / 4)


class RecipeDetectModel:
    def __init__(self, class_size, learning_rate=1e-5, training_ratio=0.8):
        self.class_size = class_size
        self.training_ratio = training_ratio
        self.learning_rate = learning_rate

    def split_train_test(self, data_set):
        train_size = int((len(data_set)) * self.training_ratio)
        train_set = data_set[:train_size]
        test_set = data_set[train_size:]

        return train_set, test_set

    def load_data_set(self):
        normalized_data_set = nd.normalized_dataset()
        normalized_data_set.make_data_set()
        data_set = normalized_data_set.get_data_set()
        training_set, test_set = self.split_train_test(data_set=data_set)

        return training_set, test_set

    @staticmethod
    def divide_data_set(data_set):
        data_set = np.array(data_set)
        image_data = data_set[:int(len(data_set)), :1].flatten()
        label_data = data_set[:int(len(data_set)), 1:].flatten()
        image_ndarray = np.empty((0, IMAGE_PIXELS))
        label_ndarray = np.empty((0, NUM_CLASSES))
        for (img, label) in zip(image_data, label_data):
            image_ndarray = np.append(image_ndarray, np.reshape(img, (1, IMAGE_PIXELS)), axis=0)
            label_ndarray = np.append(label_ndarray, np.reshape(label, (1, NUM_CLASSES)), axis=0)

        return image_ndarray, label_ndarray

    @staticmethod
    def batch_data(data_set, batch_size):
        # randomize
        data_set = random.sample(data_set, batch_size)
        return data_set

    @staticmethod
    def loss(labels_on_hot, logits):
        return -tf.reduce_sum(labels_on_hot * tf.log(logits))

    @staticmethod
    def calc_accuracy(labels_one_hot, logits):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_one_hot, 1))

        return tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    def training_optimizer(self, loss):
        return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    def inference(self, input_image_data, keep_prob):
        def weight_variable(shape):
            # 標準誤差を0.1のようにするガウス分布からランダムにWをinitialize
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            # 初期値0.1の定数としてinitialize
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        def conv2d(x, W):
            # 畳み込み層の定義、「各画像1枚1枚に対して、幅方向1、高さ方向1、チャネル方向1のストライドを行う」という意味になる。
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
            # 最大プーリングの定義、「各画像1枚1枚に対して、幅方向2、高さ方向2、チャネル方向1のストライドを行う且つ2×2の窓の最大値検出を行う」という意味になる。
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # イメージ画像(28×28)
        x_image = tf.reshape(input_image_data, [-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])

        # 3*3の畳み込みフィルターを32個用意する。[パッチサイズ,入力チャンネル数,出力チャンネル数]。
        W_conv1 = weight_variable([3, 3, 3, 32])

        # バイアス項を32個用意する。
        b_conv1 = bias_variable([32])

        # 第一畳み込みと入力の出力をReLu関数で定義
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

        # その出力を2×2で最大値検出（2×2のフィルターなので出力結果は14×14のフィルターになる）
        h_pool1 = max_pool_2x2(h_conv1)

        # 5*5の畳み込みフィルターを32種類を64個用意する。[パッチサイズ,入力チャンネル数,出力チャンネル数]。
        W_conv2 = weight_variable([3, 3, 32, 64])

        # 64個のバイアス項を用意
        b_conv2 = bias_variable([64])

        # 第一畳プーリング結果と第二層の畳み込みの結果をReLu関数で定義
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        # その出力を2×2で最大値検出（2×2のフィルターなので出力結果は7×7のフィルターになる）
        h_pool2 = max_pool_2x2(h_conv2)

        # 全結合の層 (最終的にはいる入力は7×7の縮小画像が64個あり、全結合ノードは1024と任意に設定)
        W_fc1 = weight_variable([FULL_CONNECTIVE_LAYER_SIZE * FULL_CONNECTIVE_LAYER_SIZE * 64, 1024])
        b_fc1 = bias_variable([1024])

        # 平坦化層
        h_pool2_flat = tf.reshape(h_pool2, [-1, FULL_CONNECTIVE_LAYER_SIZE * FULL_CONNECTIVE_LAYER_SIZE * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # ドロップアウトの設定
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # 全結合の層 1024(前の全結合相が1024のため)×識別クラス数を定義
        W_fc2 = weight_variable([1024, self.class_size])
        b_fc2 = bias_variable([self.class_size])

        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        return y_conv

    def train(self, session, epoch=100, batch_size=20):
        training_set, test_set = self.load_data_set()

        image_dimension = IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNEL
        images_placeholder = tf.placeholder('float', shape=(None, image_dimension))
        labels_placeholder = tf.placeholder('float', shape=(None, self.class_size))

        # dropout率を入れる仮のTensor
        keep_prob = tf.placeholder('float')

        # inference()を呼び出してモデルを作る
        inference_model = self.inference(input_image_data=images_placeholder, keep_prob=keep_prob)

        # 損失関数作成
        loss = self.loss(labels_on_hot=labels_placeholder, logits=inference_model)

        # 学習モデルのパラメーターを調整して訓練するためのオプティマイザ作成
        optimizer = self.training_optimizer(loss)

        # 精度の計算
        acc = self.calc_accuracy(labels_one_hot=labels_placeholder, logits=inference_model)

        # 変数の初期化(inferenceで利用する変数の初期化)
        session.run(tf.global_variables_initializer())

        # 指定step数だけ訓練の実行していく
        for step in range(epoch):
            # batch_size分の画像をランダム取得して画像データとラベルに分ける
            batch = self.batch_data(data_set=training_set, batch_size=batch_size)
            training_image, training_label = self.divide_data_set(batch)

            # feed_dictでplaceholderに入れるデータを指定して訓練実行
            session.run(optimizer, feed_dict={
                images_placeholder: training_image,
                labels_placeholder: training_label,
                keep_prob: 0.5})

            # 20ステップずつでログに精度を出力する(stepが出力頻度の倍数でなければ次のループへ)
            if step % 20 != 0:
                continue

            # 精度を計算する
            train_accuracy = session.run(acc, feed_dict={
                images_placeholder: training_image,
                labels_placeholder: training_label,
                keep_prob: 1.0})

            print(f'[Info]step {step}, training accuracy {train_accuracy}')

        test_image, test_label = self.divide_data_set(test_set)

        # 検証データで精度を計測
        test_accuracy = session.run(acc, feed_dict={
            images_placeholder: test_image,
            labels_placeholder: test_label,
            keep_prob: 1.0})

        print(f'[Info]test accuracy {test_accuracy}')


if __name__ == '__main__':
    rdm = RecipeDetectModel(class_size=NUM_CLASSES)
    with tf.Graph().as_default():
        session = tf.Session()
        rdm.train(session=session, epoch=1500)
