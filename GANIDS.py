from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
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

def my_one_hot(lineid):
    idImg = np.zeros([3, 16], int)
    listalpha = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'A', 'B', 'C', 'D', 'E', 'F']
    for i in range(len(lineid)):
        if i == 0:
            continue
        else:
            for j in range(len(listalpha)):
                if lineid[i] == listalpha[j]:
                    intdata = j
                    break
            for j in range(16):
                idImg[i - 1][intdata] = 1
    return idImg


def GetFreeDriveData(Attack_free_dataset_path):
    f = open(Attack_free_dataset_path)
    DataImg = []
    for line in f.readlines():
        listlinedata = line.split(',')
        lineid = listlinedata[1]
        imgid = my_one_hot(lineid)
        DataImg.append(imgid)
    return DataImg


def saveDataImg(DataImg, root_path):
    CleanData = 'CleanFreeDrivingData.txt'
    f = open(root_path + CleanData, 'w')
    for img in DataImg:
        strimg = ''
        for imgdata in img:
            for data in imgdata:
                data1 = str(data)
                strimg += data1
        f.write(strimg + '\n')
    f.close()


class GAN():
    def __init__(self):
        self.img_rows = 3
        self.img_cols = 16
        self.channels = 1

        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 50

        self.path = 'D:/文档/论文/论文/入侵检测/dataset/dataset/Sonata/CleanFreeDrivingData.txt'
        self.loadData = self.readFreeDriverData()
        # adam优化器
        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        self.generator = self.build_generator()
        gan_input = Input(shape=(self.latent_dim,))
        img = self.generator(gan_input)
        # 在训练generate的时候不训练discriminator
        self.discriminator.trainable = False
        # 对生成的假图片进行预测
        validity = self.discriminator(img)
        self.combined = Model(gan_input, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(1024, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        # ----------------------------------- #
        #   评价器，对输入进来的图片进行评价
        # ----------------------------------- #
        model = Sequential()
        # 输入一张图片
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        # 判断真伪
        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # 获得数据
        x_train = self.loadData
        print(x_train[0])
        exit(0)
        # 进行标准化
        x_train = x_train / 127.5 - 1
        x_train = np.expand_dims(x_train, axis=3)

        # 创建标签
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # 训练鉴别器
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # 训练生成器
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0

        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1

        fig.savefig("D:/文档/论文/论文/入侵检测/dataset/dataset/Sonata/GANIDS/%d.png" % epoch)
        plt.close()

    def readFreeDriverData(self):
        path = self.path
        f = open(path, 'r')
        listdata = []
        for line in f.readlines():
            line = line.split('\n')
            temp1 = []
            temp2 = []
            for i in range(len(line[0])):
                temp3 = int(line[0][i])
                if i % 16 == 0 and i != 0:
                    temp1.append(temp2)
                    temp2 = []
                temp2.append(temp3)
            temp1.append(temp2)
            listdata.append(temp1)
        ListData = np.array(listdata)
        f.close()
        return ListData


def main():
    root_path = 'D:/文档/论文/论文/入侵检测/dataset/dataset/Sonata/'
    Attack_free_dataset_path = root_path + 'FreeDrivingData_20180323_SONATA.txt'

    # 洗imgId 图像
    # DataImg = GetFreeDriveData(Attack_free_dataset_path)
    # DataImgNp = np.array(DataImg)
    # saveDataImg(DataImg,root_path)

    if not os.path.exists("D:/文档/论文/论文/入侵检测/dataset/dataset/Sonata/GANIDS"):
        os.makedirs("D:/文档/论文/论文/入侵检测/dataset/dataset/Sonata/GANIDS")

    gan = GAN()
    gan.train(epochs=20000, batch_size=256, sample_interval=200)


if __name__ == '__main__':
    main()
