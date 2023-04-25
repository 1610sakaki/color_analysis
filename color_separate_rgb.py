# ライブラリのインポート
import colorsys  # HSV計算用ライブラリ
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NUM = 1

# 画像クリップ位置スタート座標とレンジ
START_X = 0
START_Y = 0
RANGE_X = 1500
RANGE_Y = 1000

# 画像ファイルパス
IMG_PATH = "data/sample/Google-Logo.jpg"
IMG_NAME = IMG_PATH.split("/")[-1]

# K平均法クラスタ数
NUMBER_OF_CLUSTERS = 10


# 読み込む画像を指定する
class SetLoadingImage:
    def __init__(self) -> None:
        self.__read_image()

    def __read_image(self, img_path=IMG_PATH):
        try:
            read_img = cv2.imread(img_path)
            self.img = cv2.cvtColor(read_img, cv2.COLOR_BGR2RGB)  # RGB並び替え
        except cv2.error:
            print("画像読み込みエラーのため、終了します")
            sys.exit()

    @property
    def return_img(self):
        return self.img


# 画像の分析範囲を指定
class SpecifyAnalysisRange:
    def __init__(self, img) -> None:
        self.org_img = img
        self.drawed_img = np.copy(img)

    # 切り抜き範囲を描画する
    def clip(
        self,
        start_x=START_X,
        start_y=START_Y,
        draw_range_x=RANGE_X,
        draw_range_y=RANGE_Y
    ):

        self.clipped_img = self.drawed_img[
            start_y : start_y + draw_range_y,
            start_x : start_x + draw_range_x,  # (y, x)の順に記述
        ]
        # 全体画像内に描画する
        self.include_square_img = cv2.rectangle(
            img=self.org_img,  # 枠なしの画像を表示させる
            pt1=(start_x, start_y),
            pt2=(start_x + draw_range_x, start_y + draw_range_y),
            color=(255, 0, 0),
            thickness=2,
        )

        return self.clipped_img

    @property
    def overall_image(self):
        return self.include_square_img

    @property
    def cliped_image(self):
        return self.clipped_img


# K-Means法で画像を分析
class KMeansAnalyzer:
    def __init__(self, img) -> None:
        self.img = img  # 切り抜き範囲の画像を代入

        self.number_of_cluster = NUMBER_OF_CLUSTERS  # クラスタ数

    # K平均法で計算する
    def analyze(self):
        colors = self.img.reshape(-1, 3).astype(
            np.float32
        )  # 画像で使用されている色一覧。(W * H, 3) の numpy 配列。

        # K-meansアルゴリズムの収束基準を設定
        criteria = cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0
        """
        cv2.TERM_CRITERIA_MAX_ITER:反復回数が最大値に達した場合に収束判定を行うフラグ
        cv2.TERM_CRITERIA_EPS:クラスタ中心が移動する距離がしきい値以下になった場合に収束判定を行うフラグ

        criteria = cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0
        であれば、最大反復回数が10で、移動量の閾値が1.0
        """

        _, labels, rgb_value = cv2.kmeans(
            data=colors,  # クラスタリングするための入力データ
            K=self.number_of_cluster,  # クラスタ数
            bestLabels=None,  # クラスタ番号の初期値(通常はNone)
            criteria=criteria,  # アルゴリズムの収束基準
            attempts=10,  # 異なる初期値でアルゴリズムを実行する回数
            flags=cv2.KMEANS_RANDOM_CENTERS,  # クラスタリングアルゴリズムのフラグ
        )
        """
        戻り値は以下の3つ
        compactness:各点とその所属するクラスタ中心との距離の総和。
        labels:各データ点の所属するクラスタのラベル。
        centers:クラスタの中心点の座標の配列。要は画像の場合は、RGBのリストになっている。
        """
        self.labels = labels.squeeze(axis=1)  # (N, 1) -> (N,)のように要素数が1の次元を除去する
        self.rgb_value = rgb_value.astype(np.uint8)  # float32 -> uint8

        _label, self.counts = np.unique(
            self.labels, axis=0, return_counts=True
        )  # 重複したラベルを抽出し、カウント（NUMBER_OF_CLUSTERSの大きさだけラベルタイプが存在する）

        self.hsv_value = self.__rgb_to_hsv(self.rgb_value)
        self.df = self.__summarize_result(self.rgb_value, self.hsv_value, self.counts)

        return self.df

    # 計算結果をグラフ用にDataFrame化させる
    def __summarize_result(self, rgb_value, hsv_value, counts):
        df = pd.DataFrame(data=counts, columns=["counts"])
        df["R"] = rgb_value[:, 0]
        df["G"] = rgb_value[:, 1]
        df["B"] = rgb_value[:, 2]

        hsv_value = np.array(hsv_value)
        df["h"] = hsv_value[:, 0].astype(int)
        df["s"] = hsv_value[:, 1].astype(int)
        df["v"] = hsv_value[:, 2].astype(int)

        # plt用に補正
        bar_color = rgb_value / 255
        df["plt_r_color"] = bar_color[:, 0]
        df["plt_g_color"] = bar_color[:, 1]
        df["plt_b_color"] = bar_color[:, 2]

        # グラフ描画用文字列
        bar_text = list(map(str, rgb_value))
        df["text"] = bar_text

        # countsの個数順にソートして、indexを振り直す
        df = df.sort_values("counts", ascending=True).reset_index(drop=True)
        return df

    def __rgb_to_hsv(self, rgb_value):
        hsv_value_list = []
        for r, g, b in rgb_value:
            h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            h, s, v = round(h * 100), round(s * 100), round(v * 100)
            hsv_value_list.append(np.array([h, s, v]))
            print(f"RGB: ({r}, {g}, {b}) -> HSV: ({h:.2f}, {s:.2f}%, {v:.2f}%)")
        return hsv_value_list


class MakeFigure:
    def __init__(self, dataframe, image, rgb_value, labels) -> None:
        print(dataframe)
        self.df = dataframe
        self.number_of_cluster = NUMBER_OF_CLUSTERS  # クラスタ数
        self.image = image
        self.rgb_value = rgb_value
        self.labels = labels

    def output_histgram(self, ax):
        rgb_value_counts = self.df.iloc[:, 0].to_numpy().tolist()
        bar_color = self.df.iloc[:, 7:10].to_numpy().tolist()
        bar_text = self.df.iloc[:, 1:4].to_numpy().tolist()
        bar_text_list = list(map(str, bar_text))

        # ヒストグラムを表示する。
        ax.barh(
            np.arange(self.number_of_cluster),
            rgb_value_counts,
            color=bar_color,
            tick_label=bar_text_list,
        )

    def output_replaced_image(self, ax):
        # 各画素を k平均法の結果に置き換える。
        dst = self.rgb_value[self.labels].reshape(self.image.shape)
        ax.imshow(dst)


def main():
    try:
        # 読み込む画像をセットする
        img_loader = SetLoadingImage()
        original_image = img_loader.return_img

        # 分析する範囲を指定して、分析範囲を四角で囲む
        analysis_image = SpecifyAnalysisRange(original_image)
        square_range_image = analysis_image.clip()
        overall_image = analysis_image.overall_image

        # K平均法で計算する
        k_means = KMeansAnalyzer(square_range_image)
        df = k_means.analyze()

        make_figure = MakeFigure(
            dataframe=df,
            image=square_range_image,
            rgb_value=k_means.rgb_value,
            labels=k_means.labels,
        )

        # 可視化する。
        fig, [ax1, ax2, ax3, ax4] = plt.subplots(1, 4, figsize=(16, 5))
        fig.subplots_adjust(wspace=0.5)

        ax1.imshow(overall_image)
        fig.set_size_inches(18, 5)  # figを拡大する
        # 切り抜き画像を表示する。
        ax2.imshow(square_range_image)

        # ヒストグラムを表示する
        make_figure.output_histgram(ax3)

        # クラスタ数分の色値で置き換え画像を生成
        make_figure.output_replaced_image(ax4)

        ax2_title = "x:{} y:{}".format(str(START_X), str(START_Y))

        ax1.set_title("overall view " + IMG_NAME)
        ax2.set_title("cliped Image_" + ax2_title)
        ax3.set_title("histgram")
        ax4.set_title("replaced image")

        # plt.show()

    except KeyboardInterrupt:
        print("キーボード割り込み")


if __name__ == "__main__":
    main()
