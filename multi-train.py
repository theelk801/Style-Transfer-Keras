from models import *


def main():
    batch_size = 8
    image_size = 256
    style_weight = 5.0,
    content_weight = 1.0,
    denoising_weight = 1.0e-6,
    verbose = True
    cores = 8
    epochs = 1
    repeat = 2
    style_names = [
        'deer.jpg', 'soldier_chess.jpg', 'tate.jpg', 'starry_night.jpg'
    ]
    for style_name in style_names:
        for use_leaky in [False, True]:
            style_layers = ('block1_conv1', 'block2_conv1', 'block3_conv1',
                            'block4_conv1', 'block5_conv1')
            content_layer = 'block4_conv2'

            extra_name = 'config1'

            if use_leaky:
                extra_name += '_leaky'
            else:
                extra_name += '_not_leaky'

            build_and_train(
                style_name=style_name,
                style_layers=style_layers,
                content_layer=content_layer,
                batch_size=batch_size,
                image_size=image_size,
                style_weight=style_weight,
                content_weight=content_weight,
                denoising_weight=denoising_weight,
                use_leaky=use_leaky,
                verbose=verbose,
                cores=cores,
                epochs=epochs,
                repeat=repeat,
                extra_name=extra_name)

            style_layers = ('block1_conv2', 'block2_conv2', 'block3_conv3',
                            'block4_conv3')
            content_layer = 'block3_conv3'

            extra_name = 'config2'

            if use_leaky:
                extra_name += '_leaky'
            else:
                extra_name += '_not_leaky'

            build_and_train(
                style_name=style_name,
                style_layers=style_layers,
                content_layer=content_layer,
                batch_size=batch_size,
                image_size=image_size,
                style_weight=style_weight,
                content_weight=content_weight,
                denoising_weight=denoising_weight,
                use_leaky=use_leaky,
                verbose=verbose,
                cores=cores,
                epochs=epochs,
                repeat=repeat,
                extra_name=extra_name)


if __name__ == '__main__':
    main()
