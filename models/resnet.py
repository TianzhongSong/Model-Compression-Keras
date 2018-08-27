from keras.layers import GlobalAveragePooling2D, add, BatchNormalization, Activation
from keras.layers import Input
from keras.regularizers import l2
from keras.models import Model
from utils.keras_wrapped_layers import drop_connect_conv_2d, drop_connect_dense


def residual_block(x, nb_filters, strides=(1, 1), drop_prob=0., is_training=True, weight_decay=1E-4):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = drop_connect_conv_2d(x, nb_filters, (3,3),
                        kernel_initializer='he_normal',
                        padding="same",
                        strides=strides,
                        use_bias=False,
                        kernel_regularizer=l2(weight_decay),
                        drop_prob=drop_prob,
                        is_training=is_training)
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = drop_connect_conv_2d(x, nb_filters, (3, 3),
                            kernel_initializer='he_normal',
                            padding="same",
                            use_bias=False,
                            kernel_regularizer=l2(weight_decay),
                            drop_prob=drop_prob,
                             is_training=is_training)

    return x


def resnet(nb_classes, depth=20, weight_decay=1E-4, is_training=True):
    if depth == 20:
        nb_blocks = [3, 3, 3]
    elif depth == 38:
        nb_blocks = [6, 6, 6]
    elif depth == 56:
        nb_blocks = [9, 9, 9]
    else:
        raise ValueError("Only support 20, 38, 56 layers ResNet currently, but receive {}!".format(depth))

    model_input = Input(shape=(32, 32, 3))
    # init convolution
    y = drop_connect_conv_2d(model_input,
                             32,
                             (3, 3),
                             padding='same',
                             use_bias=False,
                             kernel_initializer="he_normal",
                             kernel_regularizer=l2(weight_decay),
                             drop_prob=0.2,
                             is_training=is_training)

    # stage 1
    x = residual_block(y, 32, drop_prob=0.5, is_training=is_training)
    y = add([x, y])
    for _ in range(nb_blocks[0] - 1):
        x = residual_block(y, 32, drop_prob=0.5, is_training=is_training)
        y = add([x, y])

    # stage 2
    x = residual_block(y, 64, strides=(2, 2), drop_prob=0.5, is_training=is_training)

    y = BatchNormalization(axis=-1, epsilon=1.1e-5)(y)
    y = Activation('relu')(y)
    y = drop_connect_conv_2d(y,
                             64,
                             (1, 1),
                             strides=(2, 2),
                             padding='valid',
                             use_bias=False,
                             kernel_initializer="he_normal",
                             kernel_regularizer=l2(weight_decay),
                             drop_prob=0.2,
                             is_training=is_training)
    y = add([x, y])
    for _ in range(nb_blocks[1] - 1):
        x = residual_block(y, 64, drop_prob=0.5, is_training=is_training)
        y = add([x, y])

    # stage 3
    x = residual_block(y, 128, strides=(2, 2), drop_prob=0.5, is_training=is_training)

    y = BatchNormalization(axis=-1, epsilon=1.1e-5)(y)
    y = Activation('relu')(y)
    y = drop_connect_conv_2d(y,
                             128,
                             (1, 1),
                             strides=(2, 2),
                             padding='valid',
                             use_bias=False,
                             kernel_initializer="he_normal",
                             kernel_regularizer=l2(weight_decay),
                             drop_prob=0.2,
                             is_training=is_training)
    y = add([x, y])
    for _ in range(nb_blocks[1] - 1):
        x = residual_block(y, 128, drop_prob=0.5, is_training=is_training)
        y = add([x, y])

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(y)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    x = drop_connect_dense(x,
                           nb_classes,
                           activation='softmax',
                           kernel_regularizer=l2(weight_decay),
                           bias_regularizer=l2(weight_decay))

    model = Model(inputs=model_input, outputs=x)
    return model
