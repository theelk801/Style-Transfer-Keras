from models import *


def main(batch_size=8, image_size=256, verbose=True):
    transfer = TransferModel(
        batch_size=batch_size, image_size=image_size, verbose=verbose)
    transfer.train()
    transfer.save_transfer_model()
    transfer.save_samples()


if __name__ == '__main__':
    main()
