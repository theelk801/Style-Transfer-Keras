from utils import *
from PIL import Image
from itertools import count

from keras.models import Model
from keras.applications import VGG19
from keras.layers import Flatten, Input, Conv2D, UpSampling2D, Concatenate, Subtract, Reshape, Lambda, ZeroPadding2D, Cropping2D
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
                    epochs=5):
    transfer = TransferModel(
        style_name,
        batch_size=batch_size,
        image_size=image_size,
        style_weight=style_weight,
        content_weight=content_weight,
        denoising_weight=denoising_weight,
        verbose=verbose)
    transfer.train(cores=cores, epochs=epochs)
    transfer.save_transfer_model()
    transfer.save_samples()


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
            x, 128, (3, 3), name='transfer', name_index=next(index_gen))

        x = UpSampling2D((2, 2), name='upsampling_2')(x)
        x = conv_act_norm(
            x, 64, (3, 3), name='transfer', name_index=next(index_gen))
        x = conv_act_norm(
            x, 32, (9, 9), name='transfer', name_index=next(index_gen))

        x = Conv2D(
            3, (1, 1),
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
        index_gen = count(1)
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
        return self.transfer_train.fit_generator(
            self.generator,
            use_multiprocessing=True,
            workers=cores,
            epochs=epochs,
            verbose=self.verbose)

    def save_transfer_model(self):
        self.transfer_net.save(
            f'./data/models/{self.style_name}_transfer_model.h5')

    def save_samples(self):
        for key in self.sample_ims.keys():
            im = self.transfer_net.predict(
                np.expand_dims(self.sample_ims[key], axis=0))[0]
            im = Image.fromarray(np.uint8(255 * im))
            im.save(f'./data/output/{self.style_name[:-4]}_{key}.jpg')
