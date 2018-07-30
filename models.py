import os
from utils import *
from PIL import Image
from itertools import count

from keras.models import Model
from keras.applications import VGG19
from keras.layers import Layer, Flatten, Input, Conv2D, UpSampling2D, Concatenate, Subtract, Reshape, Lambda, ZeroPadding2D, Cropping2D, LeakyReLU
from keras_contrib.layers import InstanceNormalization
from keras import backend as K

K.set_image_data_format('channels_last')


def build_and_train(style_name,
                    batch_size=8,
                    image_size=256,
                    style_weight=5.0,
                    content_weight=1.0,
                    denoising_weight=1.0e-6,
                    verbose=True,
                    cores=8,
                    epochs=5,
                    repeat=1):
    transfer = TransferModel(
        style_name,
        batch_size=batch_size,
        image_size=image_size,
        style_weight=style_weight,
        content_weight=content_weight,
        denoising_weight=denoising_weight,
        verbose=verbose)
    for _ in range(repeat):
        transfer.train(cores=cores, epochs=epochs)
        transfer.save_transfer_model()
        transfer.save_samples()


def conv_act_norm(inp,
                  kernels,
                  conv_window,
                  strides=(1, 1),
                  padding='same',
                  alpha=0.02,
                  name=None,
                  name_index=None):
    conv_name = act_name = norm_name = None
    if name is not None and name_index is not None:
        conv_name = name + f'_conv_{name_index}'
        act_name = name + f'_act_{name_index}'
        norm_name = name + f'_norm_{name_index}'
    x = Conv2D(
        kernels, conv_window, strides=strides, padding=padding,
        name=conv_name)(inp)
    x = LeakyReLU(alpha, name=act_name)(x)
    x = InstanceNormalization(axis=3, name=norm_name)(x)
    return x


def l2_loss(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1) / 2


class GramMatrix(Layer):
    def __init__(self, **kwargs):
        super(GramMatrix, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GramMatrix, self).build(input_shape)

    def call(self, inputs, **kwargs):
        temp = K.batch_dot(
            inputs, K.permute_dimensions(inputs, (0, 2, 1)), axes=[1, 2])
        b, hw, c = temp.get_shape()
        return temp / int(hw * c)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1], input_shape[-1]


class TransferModel:
    STYLE_LAYERS = ('block1_conv2', 'block2_conv2', 'block3_conv3',
                    'block4_conv3')
    CONTENT_LAYER = 'block3_conv3'
    sample_im_names = ['mountains', 'family', 'city', 'dogs']
    style_dir = './data/styles/'
    train_dir = './data/contents/resized/'
    sample_dir = './data/examples/'

    def __init__(self,
                 style_name,
                 batch_size=8,
                 image_size=256,
                 style_weight=5.0,
                 content_weight=1.0,
                 denoising_weight=1.0e-6,
                 verbose=True):
        self.batch_size = batch_size
        self.image_size = image_size
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.denoising_weight = denoising_weight
        self.verbose = verbose
        self.image_shape = (image_size, image_size, 3)
        self.style_name = style_name

        self.vgg = VGG19(include_top=False)
        self.inp = Input(self.image_shape)
        self.epochs_trained = 0

        self.transfer_net = self._create_transfer_net()
        self.style_model = self._create_style_model()
        self.content_model = self._create_content_model()
        self.denoising_model = self._create_denoising_model()
        self.transfer_train = self._create_transfer_train()

        self.sample_ims = get_samples(self.sample_dir, self.sample_im_names)

        self.style_image = open_style_image(self.style_name, self.style_dir,
                                            self.image_size, self.verbose)
        self.style_features = self.style_model.predict(self.style_image)
        self.img_dir = os.listdir(self.train_dir)
        self.generator = DataGenerator(
            img_dir=self.img_dir,
            style_features=self.style_features,
            content_model=self.content_model,
            train_path=self.train_dir,
            batch_size=self.batch_size)

    def _create_transfer_net(self):
        index_gen = count(1)
        inp = Input((None, None, 3))
        x = inp

        x = conv_act_norm(
            x, 32, (9, 9), name='transfer', name_index=next(index_gen))
        x = conv_act_norm(
            x,
            64, (3, 3),
            strides=(2, 2),
            name='transfer',
            name_index=next(index_gen))
        x = conv_act_norm(
            x,
            128, (3, 3),
            strides=(2, 2),
            name='transfer',
            name_index=next(index_gen))

        for _ in range(5):
            temp = conv_act_norm(
                x, 128, (3, 3), name='transfer', name_index=next(index_gen))
            temp = conv_act_norm(
                temp, 128, (3, 3), name='transfer', name_index=next(index_gen))
            x = Concatenate(axis=3)([x, temp])

        x = UpSampling2D((2, 2), name='upsampling_1')(x)

        x = conv_act_norm(
            x, 256, (3, 3), name='transfer', name_index=next(index_gen))
        x = conv_act_norm(
            x, 128, (3, 3), name='transfer', name_index=next(index_gen))

        x = UpSampling2D((2, 2), name='upsampling_2')(x)

        x = conv_act_norm(
            x, 64, (3, 3), name='transfer', name_index=next(index_gen))
        x = conv_act_norm(
            x, 32, (3, 3), name='transfer', name_index=next(index_gen))

        x = Conv2D(
            3, (9, 9),
            padding='same',
            activation='tanh',
            name=f'transfer_conv_{next(index_gen)}')(x)
        x = Lambda(lambda t: (t + 1) / 2, name='transfer')(x)

        transfer_net = Model(inp, x)
        if self.verbose:
            print('Transfer model built')
            transfer_net.summary()
        return transfer_net

    def _create_style_model(self):
        style_models = []
        for j, layer_name in enumerate(self.STYLE_LAYERS):
            x = self.inp
            for i, l in enumerate(self.vgg.layers):
                if i != 0:
                    x = l(x)
                    if l.name == layer_name:
                        break
            x = Reshape(
                ((self.image_shape[0] * self.image_shape[1]) // (4**j),
                 64 * (2**j)),
                name=f'style_reshape_{j}')(x)
            x = GramMatrix(name=f'style_gram_{j}')(x)
            x = Flatten(name=f'style_flatten_{j}')(x)
            style_models += [x]

        style_model = Model(
            self.inp, Concatenate(name='style_concatenate')(style_models))
        style_model.trainable = False
        style_model.name = 'style_model'
        if self.verbose:
            print('Style model built')
            style_model.summary()
        return style_model

    def _create_content_model(self):
        x = self.inp
        for i, l in enumerate(self.vgg.layers):
            if i != 0:
                x = l(x)
                if l.name == self.CONTENT_LAYER:
                    break

        x = Flatten(name='content_flatten')(x)

        content_model = Model(self.inp, x)
        content_model.trainable = False

        x = Subtract(name='content_subtract')([
            content_model(self.inp),
            content_model(self.transfer_net(self.inp))
        ])
        content_model = Model(self.inp, x)
        content_model.name = 'content_model'
        if self.verbose:
            print('Content model built')
            content_model.summary()
        return content_model

    def _create_denoising_model(self):
        x = Subtract(name='denoising_subtract_1')([
            ZeroPadding2D(padding=((0, 0), (0, 1)),
                          name='padding_1')(self.inp),
            ZeroPadding2D(padding=((0, 0), (1, 0)), name='padding_2')(self.inp)
        ])
        x = Cropping2D(cropping=((0, 0), (1, 1)), name='cropping_1')(x)
        y = Subtract(name='denoising_subtract_2')([
            ZeroPadding2D(padding=((0, 1), (0, 0)),
                          name='padding_3')(self.inp),
            ZeroPadding2D(padding=((1, 0), (0, 0)), name='padding_4')(self.inp)
        ])
        y = Cropping2D(cropping=((1, 1), (0, 0)), name='cropping_2')(y)
        x = Flatten(name='denoising_flatten_1')(x)
        y = Flatten(name='denoising_flatten_2')(y)
        x = Concatenate(name='denoising_concatenate')([x, y])
        x = Lambda(lambda t: 255 * t, name='denoising_scale')(x)
        denoising_model = Model(self.inp, x)
        denoising_model.name = 'denoising_model'
        if self.verbose:
            print('Denoising model built')
            denoising_model.summary()
        return denoising_model

    def _create_transfer_train(self):
        transfer_train = Model(self.inp, [
            self.style_model(self.transfer_net(self.inp)),
            self.content_model(self.inp),
            self.denoising_model(self.transfer_net(self.inp))
        ])
        if self.verbose:
            print('Full training model built')
            transfer_train.summary()
        transfer_train.compile(
            'adam',
            loss=l2_loss,
            loss_weights={
                'style_model': self.style_weight,
                'content_model': self.content_weight,
                'denoising_model': self.denoising_weight
            })
        return transfer_train

    def train(self, cores=8, epochs=5):
        history = self.transfer_train.fit_generator(
            self.generator,
            use_multiprocessing=True,
            workers=cores,
            epochs=epochs,
            verbose=self.verbose)
        self.epochs_trained += epochs
        return history

    def save_transfer_model(self):
        self.transfer_net.save(
            f'./data/models/{self.style_name}_transfer_model_{self.epochs_trained}.h5'
        )
        if self.verbose:
            print(f'Model saved after {self.epochs_trained} epochs')

    def save_samples(self):
        for key in self.sample_ims.keys():
            im = self.transfer_net.predict(
                np.expand_dims(self.sample_ims[key], axis=0))[0]
            im = Image.fromarray(np.uint8(255 * im))
            im.save(
                f'./data/output/{self.style_name[:-4]}_{key}_{self.epochs_trained}.jpg'
            )
        if self.verbose:
            print(f'Samples saved after {self.epochs_trained} epochs')
