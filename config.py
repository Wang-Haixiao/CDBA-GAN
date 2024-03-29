# -*- coding: utf-8 -*-


class Config():

    def __init__(self):

        #
        # train 関連のパラメータです。
        #
        # 学習のエポック数。
        self.EPOCH = 2000
        # 1 エポックあたりの iteration 数。
        self.ITER_PER_EPOCH = 100
        # 学習を途中から再開する時用のフラグです。
        self.RESUME_TRAIN = False
        # result ディレクトリ下のどのディレクトリの結果から再開するか。
        self.RESUME_FROM = "180912_1031"
        # 何iteration学習した結果から再開するか。
        self.COUNTER = 245000
        # critic 数。unrolled GAN や WGAN を使う時用に。
        self.NUM_CRITICS = 1
        self.num_classes = 5
        self.LATENT_DIM = 128

        # ミニバッチサイズです。
        # self.BATCH_SIZE = 32
        self.BATCH_SIZE = 1
        # 学習率です。
        self.D_LEARNING_RATE = 0.0004
        self.G_LEARNING_RATE = 0.0001
        # Adam optimizerのハイパーパラメータです。
        self.BETA_1 = 0.0
        self.BETA_2 = 0.9
        # cycle consitency lossの係数です。
        self.LAMBDA = 10
        # Discriminator に Residual Block を使用するフラグです。
        self.USE_RES = True
        # Residual Blockの数です。
        self.NUMBER_RESIDUAL_BLOCKS = 3

        # 学習結果を出力するディレクトリです。
        self.RESULT_DIR = "./result/"
        # 訓練データを格納するディレクトリです。
        self.DATA_DIR = "./data/"

        # 学習時の入力画像サイズです。 (height, width, channel)
        # self.IMAGE_SHAPE = (64, 64, 3)
        self.IMAGE_SHAPE = (256, 192, 1)
        # 訓練データのファイル拡張子です。
        # TODO: jpg、pngが混在していても使用できるようにする。
        self.DATA_EXT = "*.png"
        # 訓練データを格納しているディレクトリの名前です。
        # self.DATASET = "all-dogs"
        self.DATASET = "sonar_new"

        # 使用するモデルのweightファイルのパスです。
        # self.MODEL_FILE_PATH = "./result/191226_0010/weights/sonar_new/generator190000.hdf5"

        # self.MODEL_FILE_PATH = "./result/nonoise200528_1529/weights/sonar_new/generator190000.hdf5"
        self.MODEL_FILE_PATH = "./result/200713_2011/weights/sonar_new/generator190000.hdf5"
    def output_config(self, path):
        """
        config の内容を txt として出力するメソッドです。
        :param path: str
            出力先のパス。
        """
        output_str_list = []
        output_str_list.append("Epoch:{}".format(self.EPOCH))
        output_str_list.append("Iteration per Epoch:{}".format(self.ITER_PER_EPOCH))
        output_str_list.append("Number of Critics:{}".format(self.NUM_CRITICS))
        output_str_list.append("MiniBatch Size:{}".format(self.BATCH_SIZE))
        output_str_list.append("num_classes:{}".format(self.num_classes))
        output_str_list.append("Discriminator's Learning Rate:{}".format(self.D_LEARNING_RATE))
        output_str_list.append("Generator's Learning Rate:{}".format(self.G_LEARNING_RATE))
        output_str_list.append("Beta 1:{}".format(self.BETA_1))
        output_str_list.append("Beta 2:{}".format(self.BETA_2))
        output_str_list.append("Lambda for gradient penalty:{}".format(self.LAMBDA))
        output_str_list.append("Use Residual Network:{}".format(self.USE_RES))
        output_str_list.append("Result Output Directory:{}".format(self.RESULT_DIR))
        output_str_list.append("Input Data Directory:{}".format(self.DATA_DIR))
        output_str_list.append("Input Shape:({},{},{})".format(self.IMAGE_SHAPE[0],self.IMAGE_SHAPE[1],self.IMAGE_SHAPE[2]))
        output_str_list.append("Data Extent:{}".format(self.DATA_EXT))
        output_str_list.append("DataSet:{}".format(self.DATASET))

        with open(path, mode="w") as f:
            config_txt = "\n".join(output_str_list)
            f.write(config_txt)



