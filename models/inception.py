from keras.layers import GlobalAveragePooling2D, concatenate, BatchNormalization, Activation
from keras.layers import Input, AveragePooling2D, MaxPooling2D, Conv2D, Dense
from keras.regularizers import l2
from keras.models import Model


def conv2d_factory(x, nb_filters, kernel_size=(3, 3), weight_decay=1E-4):
    x = Conv2D(nb_filters, kernel_size,
             kernel_initializer='he_normal',
             padding="same",
             use_bias=False,
             kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def inception_v3(nb_classes):
    weight_decay = 1E-4
    model_input = Input(shape=(32, 32, 3))

    # init convolution
    x = conv2d_factory(model_input, 32)
    x = conv2d_factory(x, 32)

    # stage 1 32x32x64
    for _ in range(2):
        branch1x1 = conv2d_factory(x, 32, (1, 1))

        branch5x5 = conv2d_factory(x, 32, (1, 1))
        branch5x5 = conv2d_factory(branch5x5, 16, (5, 5))

        branch3x3 = conv2d_factory(x, 32, (1, 1))
        branch3x3 = conv2d_factory(branch3x3, 32, (3, 3))
        branch3x3 = conv2d_factory(branch3x3, 32, (3, 3))

        branch_pool = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_factory(branch_pool, 32, (1, 1))

        x = concatenate([branch1x1, branch5x5, branch3x3, branch_pool], axis=-1)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # stage 2 16x16x128
    for _ in range(2):
        branch1x1 = conv2d_factory(x, 64, (1, 1))

        branch5x5 = conv2d_factory(x, 64, (1, 1))
        branch5x5 = conv2d_factory(branch5x5, 64, (5, 1))
        branch5x5 = conv2d_factory(branch5x5, 64, (1, 5))

        branch7x7 = conv2d_factory(x, 64, (1, 1))
        branch7x7 = conv2d_factory(branch7x7, 64, (7, 1))
        branch7x7 = conv2d_factory(branch7x7, 64, (1, 7))
        branch7x7 = conv2d_factory(branch7x7, 64, (7, 1))
        branch7x7 = conv2d_factory(branch7x7, 64, (1, 7))

        branch_pool = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_factory(branch_pool, 64, (1, 1))

        x = concatenate([branch1x1, branch5x5, branch7x7, branch_pool], axis=-1)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # stage 3 8x8x256
    for _ in range(2):
        branch1x1 = conv2d_factory(x, 128, (1, 1))

        branch5x5 = conv2d_factory(x, 128, (1, 1))
        branch5x5 = conv2d_factory(branch5x5, 128, (5, 1))
        branch5x5 = conv2d_factory(branch5x5, 128, (1, 5))

        branch3x3_2 = conv2d_factory(x, 128, (1, 1))
        branch3x3_2 = conv2d_factory(branch3x3_2, 128, (3, 3))
        branch3x3_2 = conv2d_factory(branch3x3_2, 128, (3, 1))
        branch3x3_2 = conv2d_factory(branch3x3_2, 128, (1, 3))

        branch_pool = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_factory(branch_pool, 128, (1, 1))

        x = concatenate([branch1x1, branch5x5, branch3x3_2, branch_pool], axis=-1)

    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes, activation='softmax',
               kernel_regularizer=l2(weight_decay),
               bias_regularizer=l2(weight_decay))(x)

    model = Model(inputs=model_input, outputs=x)
    return model
