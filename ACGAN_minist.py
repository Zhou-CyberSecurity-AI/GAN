from __future__ import print_function, division
import tensorflow as tf
from keras.datasets import mnist
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
import tensorflow as tf
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
warnings.filterwarnings("ignore")


class ACGAN():
    def __init__(self):

        # 输入shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        # 分十类
        self.num_classes = 10
        self.latent_dim = 100
        # adam优化器
        optimizer = Adam(0.0002, 0.5)
        # 判别模型
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses,
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # 生成模型
        self.generator = self.build_generator()

        # conbine是生成模型和判别模型的结合
        # 判别模型的trainable为False
        # 用于训练生成模型
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        self.discriminator.trainable = False

        valid, target_label = self.discriminator(img)

        self.combined = Model([noise, label], [valid, target_label])
        self.combined.compile(loss=losses,
                              optimizer=optimizer)

    def build_generator(self):
        model = Sequential()

        # n维输入转化成特征层 DCGAN
        model.add(Dense(32 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 32)))

        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        # 扩张他的长和宽
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D())

        # 28*28*64
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()
        # 生成图片的标签
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')

        # Embedding 层可以将正整数转化为固定尺寸的稠密向量
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        # ----------------------------------- #
        #   评价器，对输入进来的图片进行评价
        # ----------------------------------- #
        model = Sequential()
        # ACGAN

        # 28,28,1 ----->  14,14,32
        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        # 14,14,16 ----->  8,8,32
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        # 8,8,32 ----->  4,4,64

        model.add(ZeroPadding2D(((0, 1), (0, 1))))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(GlobalAveragePooling2D())
        model.summary()
        img = Input(shape=self.img_shape)

        features = model(img)
        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes, activation="softmax")(features)
        return Model(img, [validity, label])

    def train(self, epochs, batch_size=128, sample_interval=50):

        # 获得数据
        (x_train, y_train), (_, _) = mnist.load_data()

        # 进行标准化
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5
        x_train = np.expand_dims(x_train, axis=3)
        y_train = y_train.reshape(-1, 1)
        print(y_train[0])
        Dloss = []
        Gloss = []
        Dacc = []
        Gacc = []
        # 创建标签
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # 训练鉴别器
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs, labels = x_train[idx], y_train[idx]

            # ---------------------- #
            #   生成正态分布的输入
            # ---------------------- #
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            sampled_labels = np.random.randint(0, 10, (batch_size, 1))
            gen_imgs = self.generator.predict([noise, sampled_labels])

            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, sampled_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # 训练生成器
            g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])
            print("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (
            epoch, d_loss[0], 100 * d_loss[3], 100 * d_loss[4], g_loss[0]))
            Dloss.append(d_loss[0])
            Gloss.append(g_loss[0])
            Dacc.append(100 * d_loss[3])
            Gacc.append(100 * d_loss[4])
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
        self.image(Dloss, Gloss, Dacc, Gacc)


    def image(self,Dloss,Gloss,Dacc,Gacc):
        epochs = range(1, len(Dloss) + 1)
        plt.plot(epochs, Dloss, linestyle="--", color='red', label='Discrimination loss')
        plt.plot(epochs, Gloss, linestyle="-.", color='blue', label='Generator loss')
        plt.legend()
        # plt.text(x, y + 0.05, '%.0f' % y, ha='center', va='bottom', fontsize=11)
        ax = plt.gca()
        ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
        ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示

        plt.title('Discrimination and Generator loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

        plt.clf()

        plt.plot(epochs, Dacc, linestyle="--", color='red', label='Discrimination acc', marker='o')
        plt.plot(epochs, Gacc, linestyle="-.", color='blue', label='Generator acc', marker='>')
        plt.legend()

        ax = plt.gca()
        ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
        ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示
        plt.title('Discrimination and Generator accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()
    def sample_images(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.arange(0, 10).reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels])
        gen_imgs = 0.5 * gen_imgs + 0.5
        print(gen_imgs.shape)
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                print(gen_imgs[cnt,:,:,0].shape)
                axs[i, j].set_title("Digit: %d" % sampled_labels[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("D:/文档/论文/论文/入侵检测/dataset/dataset/Sonata/ACGAN" + "/%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                       "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")


if __name__ == '__main__':
    if not os.path.exists("D:/文档/论文/论文/入侵检测/dataset/dataset/Sonata/ACGAN"):
        os.makedirs("D:/文档/论文/论文/入侵检测/dataset/dataset/Sonata/ACGAN")

    gan = ACGAN()
    gan.train(epochs=1000, batch_size=256, sample_interval=200)