import numpy as np

from keras import backend as K
from keras.engine.topology import Layer
from keras.utils import Sequence
from PIL import Image

K.set_image_data_format('channels_last')


class Gram_Matrix(Layer):
    def __init__(self, weight=1, **kwargs):
        self.weight = weight**0.5
        super(Gram_Matrix, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Gram_Matrix, self).build(input_shape)

    def call(self, x):
        temp = K.batch_dot(x, K.permute_dimensions(x, (0, 2, 1)), axes=[1, 2])
        b, hw, c = temp.get_shape()
        return self.weight * temp / int(hw * c)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1], input_shape[-1])


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


def open_style_image(image_name, style_dir, image_size):
    style_image = Image.open(style_dir + image_name)
    style_image = np.array(style_image.resize((image_size, image_size)))
    style_image = style_image.reshape((1, image_size, image_size, 3)) / 255
    return style_image


class DataGenerator(Sequence):
    def __init__(self,
                 imdir,
                 style_features,
                 content_model,
                 train_path,
                 h=256,
                 w=256,
                 c=3,
                 batch_size=4):
        'Initialization'
        self.imdir = imdir
        self.train_path = train_path
        self.style_features = np.repeat(style_features, batch_size, axis=0)
        self.content_zeroes = np.zeros((batch_size,
                                        content_model.output_shape[1]))
        self.h = h
        self.w = w
        self.c = 3
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return len(self.imdir) // self.batch_size

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]
        im_temp = [self.imdir[k] for k in indexes]
        X, y = self.__data_generation(im_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.imdir))
        np.random.shuffle(self.indexes)

    def __data_generation(self, im_temp):
        X = np.empty((self.batch_size, self.h, self.w, self.c))

        for i, im in enumerate(im_temp):
            temp = np.array(Image.open(self.train_path + im)) / 255
            if len(temp.shape) == 2:
                temp = np.repeat(np.expand_dims(temp, axis=2), 3, axis=2)
            X[i, ] = temp

        return X, [self.style_features, self.content_zeroes]
