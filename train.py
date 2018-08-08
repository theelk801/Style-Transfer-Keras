from sys import argv
from models import *


def main(args):
    batch_size = 8
    image_size = 256
    style_weight = 5.0,
    content_weight = 1.0,
    denoising_weight = 1.0e-6,
    verbose = True
    style_layers = ('block1_conv2', 'block2_conv2', 'block3_conv3',
                    'block4_conv3')
    content_layer = 'block3_conv3'
    cores = 8
    epochs = 1
    repeat = 20
    if len(args) > 1:
        style_name = args[1]
    else:
        style_name = 'style.jpg'
    build_and_train(
        style_name=style_name,
        style_layers=style_layers,
        content_layer=content_layer,
        batch_size=batch_size,
        image_size=image_size,
        style_weight=style_weight,
        content_weight=content_weight,
        denoising_weight=denoising_weight,
        verbose=verbose,
        cores=cores,
        epochs=epochs,
        repeat=repeat)


if __name__ == '__main__':
    main(argv)
