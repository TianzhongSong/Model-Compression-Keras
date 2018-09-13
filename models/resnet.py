from keras.layers import Conv2D, Dense, Activation, BatchNormalization, Input
from keras.layers import GlobalAveragePooling2D, add
from keras.regularizers import l2
from keras.models import Model


def residual_block(x, nb_filters, strides=(1, 1), weight_decay=1E-4):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filters, (3,3),
                kernel_initializer='he_normal',
                padding="same",
                strides=strides,
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filters, (3, 3),
                kernel_initializer='he_normal',
                padding="same",
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(x)

    return x


def resnet(nb_classes, depth=20, wide_factor=1, weight_decay=1E-4):
    if depth == 20:
        nb_blocks = [3, 3, 3]
    elif depth == 28:
        nb_blocks = [4, 5, 4]
    elif depth == 40:
        nb_blocks = [6, 7, 6]
    elif depth == 56:
        nb_blocks = [9, 9, 9]
    elif depth == 110:
        nb_blocks = [18, 18, 18]
    else:
        raise ValueError("Only support 20, 28, 40, 56, 110 layers ResNet currently, but receive {}!".format(depth))
    nb_filters = [16 * wide_factor, 32 * wide_factor, 64 * wide_factor]
    model_input = Input(shape=(32, 32, 3))
    # init convolution
    y = Conv2D(nb_filters[0],
             (3, 3),
             padding='same',
             use_bias=False,
             kernel_initializer="he_normal",
             kernel_regularizer=l2(weight_decay))(model_input)

    # stage 1
    x = residual_block(y, nb_filters[0])
    y = add([x, y])
    for _ in range(nb_blocks[0] - 1):
        x = residual_block(y, nb_filters[0])
        y = add([x, y])

    # stage 2
    x = residual_block(y, nb_filters[1], strides=(2, 2))

    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(nb_filters[1],
             (1, 1),
             strides=(2, 2),
             padding='valid',
             use_bias=False,
             kernel_initializer="he_normal",
             kernel_regularizer=l2(weight_decay))(y)

    y = add([x, y])
    for _ in range(nb_blocks[1] - 1):
        x = residual_block(y, nb_filters[1])
        y = add([x, y])

    # stage 3
    x = residual_block(y, nb_filters[2], strides=(2, 2))

    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(nb_filters[2],
             (1, 1),
             strides=(2, 2),
             padding='valid',
             use_bias=False,
             kernel_initializer="he_normal",
             kernel_regularizer=l2(weight_decay))(y)
    y = add([x, y])
    for _ in range(nb_blocks[1] - 1):
        x = residual_block(y, nb_filters[2])
        y = add([x, y])

    x = BatchNormalization()(y)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes,
               activation='softmax',
               kernel_regularizer=l2(weight_decay),
               bias_regularizer=l2(weight_decay))(x)

    model = Model(inputs=model_input, outputs=x)
    return model
