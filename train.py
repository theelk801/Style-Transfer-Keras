from sys import argv
from models import *


def main(argv):
    batch_size = 8
    image_size = 256
    style_weight=5.0,
    content_weight=1.0,
    denoising_weight=1.0e-6,
    verbose = True
    cores = 8
    epochs = 5
    if len(argv) > 1:
        style_name = argv[1]
    else:
        style_name = 'style.jpg'
    build_and_train(
        style_name=style_name,
        batch_size=batch_size,
        image_size=image_size,
        style_weight=style_weight,
        content_weight=content_weight,
        denoising_weight=denoising_weight,
        verbose=verbose,
        cores=cores,
        epochs=epochs)


if __name__ == '__main__':
    main(argv)
