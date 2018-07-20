from models import *


def build_and_train(style_name, batch_size=8, image_size=256, verbose=True):
    transfer = TransferModel(
        style_name,
        batch_size=batch_size,
        image_size=image_size,
        verbose=verbose)
    transfer.train()
    transfer.save_transfer_model()
    transfer.save_samples()


def main():
    batch_size = 8
    image_size = 256
    verbose = True
    build_and_train(
        'style.jpg',
        batch_size=batch_size,
        image_size=image_size,
        verbose=verbose)


if __name__ == '__main__':
    main()
