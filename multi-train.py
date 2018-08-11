from models import *


def main():
    batch_size = 8
    image_size = 256
    style_weight = 150.0,
    content_weight = 7.5,
    denoising_weight = 1.0,
    verbose = True
    cores = 8
    epochs = 1
    repeat = 2
    style_names = [
        'deer.jpg', 'soldier_chess.jpg', 'tate.jpg', 'starry_night.jpg'
    ]
    for style_name in style_names:
        build_and_train(
            style_name=style_name,
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
    main()
