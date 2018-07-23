import os
import numpy as np

from keras import backend as K
from keras.layers import Layer, LeakyReLU, Conv2D
from keras.utils import Sequence
from keras_contrib.layers import InstanceNormalization
from PIL import Image

K.set_image_data_format('channels_last')


class GramMatrix(Layer):
    def __init__(self, weight=1, **kwargs):
        self.weight = weight**0.5
        super(GramMatrix, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GramMatrix, self).build(input_shape)

    def call(self, inputs, **kwargs):
        temp = K.batch_dot(
            inputs, K.permute_dimensions(inputs, (0, 2, 1)), axes=[1, 2])
        b, hw, c = temp.get_shape()
        return self.weight * temp / int(hw * c)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1], input_shape[-1]


def l2_loss(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1) / 2


def conv_act_norm(inp,
                  kernels,
                  conv_window,
                  strides=(1, 1),
                  padding='same',
                  alpha=0.02):
    x = Conv2D(kernels, conv_window, strides=strides, padding=padding)(inp)
    x = LeakyReLU(alpha)(x)
    x = InstanceNormalization(axis=3)(x)
    return x


def open_im(path, square=False):
    im = Image.open(path)
    if square:
        m = max(im.height, im.width)
        im = im.resize((round(256 * im.width / m), round(256 * im.height / m)))
        blank = Image.new('RGB', (256, 256), color=(255, 255, 255))
        blank.paste(
            im, box=(int((256 - im.width) / 2), int((256 - im.height) / 2)))
        im = np.array(blank)
    else:
        im = np.array(im)
    if len(im.shape) == 2:
        return np.array(3 * [im]).transpose((1, 2, 0))
    return im


def get_samples(sample_dir, sample_im_names):
    sample_ims = dict()
    for sample_name in sample_im_names:
        sample_ims[
            sample_name] = open_im(sample_dir + sample_name + '.jpg') / 255
    return sample_ims


def open_style_image(style_name, style_dir, image_size, verbose=True):
    style_image = Image.open(style_dir + style_name)
    style_image = np.array(style_image.resize((image_size, image_size)))
    style_image = style_image.reshape((1, image_size, image_size, 3)) / 255
    if verbose:
        print(f'Opened style image: {style_name}')
    return style_image


class DataGenerator(Sequence):
    def __init__(self,
                 img_dir,
                 style_features,
                 content_model,
                 train_path,
                 h=256,
                 w=256,
                 c=3,
                 batch_size=4):
        self.img_dir = img_dir
        self.train_path = train_path
        self.style_features = np.repeat(style_features, batch_size, axis=0)
        self.content_zeroes = np.zeros((batch_size,
                                        content_model.output_shape[1]))
        self.denoising_zeroes = np.zeros((batch_size,
                                          (((h - 1) * w) + (h * (w - 1))) * c))
        self.h = h
        self.w = w
        self.c = c
        self.batch_size = batch_size
        self.indexes = np.arange(len(img_dir))
        self.on_epoch_end()

    def __len__(self):
        return len(self.img_dir) // self.batch_size

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]
        im_temp = [self.img_dir[k] for k in indexes]
        x, y = self.__data_generation(im_temp)

        return x, y

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __data_generation(self, im_temp):
        x = np.empty((self.batch_size, self.h, self.w, self.c))

        for i, im in enumerate(im_temp):
            temp = np.array(Image.open(self.train_path + im)) / 255
            if len(temp.shape) == 2:
                temp = np.repeat(np.expand_dims(temp, axis=2), 3, axis=2)
            x[i, ] = temp

        return x, [
            self.style_features, self.content_zeroes, self.denoising_zeroes
        ]
