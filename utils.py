import numpy as np
import scipy
import scipy.misc

from keras.utils import Sequence


def open_im(image_path, img_size=None):
    img = scipy.misc.imread(image_path)
    if (len(img.shape) != 3) or (img.shape[2] != 3):
        img = np.dstack((img, img, img))

    if (img_size is not None):
        img = scipy.misc.imresize(img, img_size)

    img = img.astype("float32")
    return img


def get_samples(sample_dir, sample_im_names):
    sample_ims = dict()
    for sample_name in sample_im_names:
        sample_ims[
            sample_name] = open_im(sample_dir + sample_name + '.jpg') / 255
    return sample_ims


def save_image(img, path):
    scipy.misc.imsave(path, np.clip(img, 0, 255).astype(np.uint8))


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
            x[i, ] = open_im(self.train_path + im, (self.h, self.w)) / 255

        return x, [
            self.style_features, self.content_zeroes, self.denoising_zeroes
        ]
