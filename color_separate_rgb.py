# ライブラリのインポート
import colorsys  # HSV計算用ライブラリ
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 画像のパス、名前、切り抜く範囲を指定する
IMG_PATH = "data/sample/Google-Logo.jpg"
# IMG_PATH = "data/sample/sea-free-photo5.jpg"
# IMG_PATH = "data/sample/fresh-fruits-2305192_960_720.jpg"
IMG_NAME = IMG_PATH.split("/")[-1]
START_X, START_Y = 0, 0
RANGE_X, RANGE_Y = 1500, 1000
NUMBER_OF_CLUSTERS = 5


# 読み込む画像を指定する
class SetLoadingImage:
    def __init__(self, img_path=IMG_PATH):
        self.img = self.read_image(img_path)

    def read_image(self, img_path):
        try:
            read_img = cv2.imread(img_path)
            img = cv2.cvtColor(read_img, cv2.COLOR_BGR2RGB)
            return img
        except cv2.error:
            print("画像読み込みエラーのため、終了します")
            sys.exit()

    @property
    def return_img(self):
        return self.img


class SpecifyAnalysisRange:
    def __init__(self, img):
        self.org_img = img
        self.drawed_img = np.copy(img)

    def clip(
        self,
        start_x=START_X,
        start_y=START_Y,
        draw_range_x=RANGE_X,
        draw_range_y=RANGE_Y,
    ):
        self.clipped = self.drawed_img[
            start_y : start_y + draw_range_y, start_x : start_x + draw_range_x
        ]
        self.include_square_img = cv2.rectangle(
            img=self.org_img,
            pt1=(start_x, start_y),
            pt2=(start_x + draw_range_x, start_y + draw_range_y),
            color=(255, 0, 0),
            thickness=2,
        )
        return self.clipped

    @property
    def overall_image(self):
        self.clip()
        return self.include_square_img

    @property
    def clipped_image(self):
        return self.clipped


# K-Means法で画像を分析
class KMeansAnalyzer:
    def __init__(self, img):
        self.img = img  # 切り抜き範囲の画像を代入
        self.number_of_cluster = NUMBER_OF_CLUSTERS  # クラスタ数

    # K平均法で計算する
    def analyze(self) -> pd.DataFrame:
        colors = self.img.reshape(-1, 3).astype(
            np.float32
        )  # 画像で使用されている色一覧。(W * H, 3) の numpy 配列。

        # K-meansアルゴリズムの収束基準を設定
        criteria = cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0

        _, labels, rgb_value = cv2.kmeans(
            data=colors,  # クラスタリングするための入力データ
            K=self.number_of_cluster,  # クラスタ数
            bestLabels=None,  # クラスタ番号の初期値(通常はNone)
            criteria=criteria,  # アルゴリズムの収束基準
            attempts=10,  # 異なる初期値でアルゴリズムを実行する回数
            flags=cv2.KMEANS_RANDOM_CENTERS,  # クラスタリングアルゴリズムのフラグ
        )

        self.labels = labels.squeeze(axis=1)  # (N, 1) -> (N,)のように要素数が1の次元を除去する
        self.rgb_value = rgb_value.astype(np.uint8)  # float32 -> uint8

        _, self.counts = np.unique(
            self.labels, axis=0, return_counts=True
        )  # 重複したラベルを抽出し、カウント（NUMBER_OF_CLUSTERSの大きさだけラベルタイプが存在する）

        self.df = self.__summarize_result(self.rgb_value, self.counts)

        return self.df

    # 計算結果をグラフ用にDataFrame化させる
    @staticmethod
    def __summarize_result(rgb_value, counts):
        df = pd.DataFrame(data=counts, columns=["counts"])
        df["R"] = rgb_value[:, 0]
        df["G"] = rgb_value[:, 1]
        df["B"] = rgb_value[:, 2]

        # plt用に補正
        bar_color = rgb_value / 255
        df["plt_R_value"] = bar_color[:, 0]
        df["plt_G_value"] = bar_color[:, 1]
        df["plt_B_value"] = bar_color[:, 2]

        # グラフ描画用文字列
        bar_text = list(map(str, rgb_value))
        df["plt_text"] = bar_text

        # countsの個数順にソートして、indexを振り直す
        df = df.sort_values("counts", ascending=True).reset_index(drop=True)
        return df


class MakeFigure:
    def __init__(self, dataframe, overall_image, cliped_image, rgb_value, labels):
        print(dataframe)
        self.df = dataframe  # DataFrame
        self.number_of_cluster = NUMBER_OF_CLUSTERS  # クラスタ数
        self.overall_image = overall_image  # 全体画像
        self.cliped_image = cliped_image  # 切り抜き画像
        self.rgb_value = rgb_value  # RGB値
        self.labels = labels  # 図専用ラベル

    def output_overall_image(self, ax):
        ax.imshow(self.overall_image)

    def output_cliped_image(self, ax):
        ax.imshow(self.cliped_image)

    def output_histgram(self, ax):
        rgb_value_counts = (
            self.df.loc[:, ["counts"]].to_numpy().flatten().tolist()
        )  # ヒストグラム用のrgb値カウント数

        bar_color = (
            self.df.loc[:, ["plt_R_value", "plt_G_value", "plt_B_value"]]
            .to_numpy()
            .tolist()
        )  # ヒストグラム用のrgb値カウント数

        bar_text = self.df.loc[:, ["plt_text"]].to_numpy().flatten()  # ヒストグラム用x軸ラベル

        # ヒストグラムを表示する。
        ax.barh(
            np.arange(self.number_of_cluster),
            rgb_value_counts,
            color=bar_color,
            tick_label=bar_text,
        )

    def output_replaced_image(self, ax):
        # 各画素を k平均法の結果に置き換える。
        self.dst = self.rgb_value[self.labels].reshape(self.cliped_image.shape)
        ax.imshow(self.dst)


def main():
    try:
        # 読み込む画像をセットする
        img_loader = SetLoadingImage()
        original_image = img_loader.return_img

        # 分析する範囲を指定して、分析範囲を四角で囲む
        analysis_image = SpecifyAnalysisRange(original_image)
        overall_image = analysis_image.overall_image

        # K平均法で計算する
        k_means = KMeansAnalyzer(analysis_image.clipped)
        df = k_means.analyze()

        make_figure = MakeFigure(
            dataframe=df,
            cliped_image=analysis_image.clipped,
            overall_image=overall_image,
            rgb_value=k_means.rgb_value,
            labels=k_means.labels,
        )

        # 可視化する。
        fig, [ax1, ax2, ax3, ax4] = plt.subplots(1, 4, figsize=(16, 5))
        fig.subplots_adjust(wspace=0.5)

        # 全体画像を表示する
        make_figure.output_overall_image(ax1)
        # 切り抜き画像を表示する。
        make_figure.output_cliped_image(ax2)
        # ヒストグラムを表示する
        make_figure.output_histgram(ax3)
        # クラスタ数分のRGB値で置き換え画像を生成
        make_figure.output_replaced_image(ax4)

        # 各タイトル
        ax2_title = "x:{} y:{}".format(str(START_X), str(START_Y))
        ax1.set_title("overall view " + IMG_NAME)
        ax2.set_title("cliped Image_" + ax2_title)
        ax3.set_title("histgram")
        ax4.set_title("replaced image")

        # クラスタ数分のRGB値で置き換え画像のみを表示
        fig, ax = plt.subplots()
        ax.imshow(make_figure.dst)
        plt.show()

    except KeyboardInterrupt:
        print("キーボード割り込み")


if __name__ == "__main__":
    main()
