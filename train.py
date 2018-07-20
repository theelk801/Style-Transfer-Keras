from sys import argv
from models import *


def main(argv):
    batch_size = 8
    image_size = 256
    verbose = True
    if len(argv) > 1:
        style_name = argv[1]
    else:
        style_name = 'style.jpg'
    build_and_train(
        style_name=style_name,
        batch_size=batch_size,
        image_size=image_size,
        verbose=verbose)


if __name__ == '__main__':
    main(argv)
