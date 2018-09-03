from keras.layers import GlobalAveragePooling2D, concatenate, BatchNormalization, Activation
from keras.layers import Input, AveragePooling2D, MaxPooling2D
from keras.regularizers import l2
from keras.models import Model
from utils.keras_wrapped_layers import drop_connect_conv_2d, drop_connect_dense


def conv2d_factory(x, nb_filters, kernel_size=(3, 3), drop_prob=0.5,
                 is_training=True, weight_decay=1E-4):
    x = drop_connect_conv_2d(x, nb_filters, kernel_size,
                         kernel_initializer='he_normal',
                         padding="same",
                         use_bias=False,
                         kernel_regularizer=l2(weight_decay),
                         drop_prob=drop_prob,
                         is_training=is_training)
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    return x


def inception_v3(nb_classes, is_training=True):
    weight_decay = 1E-4
    model_input = Input(shape=(32, 32, 3))

    # init convolution
    x = conv2d_factory(model_input, 32, drop_prob=0.0, is_training=is_training)
    x = conv2d_factory(x, 32, drop_prob=0.0, is_training=is_training)

    # stage 1 32x32x64
    for _ in range(2):
        branch1x1 = conv2d_factory(x, 32, (1, 1), is_training=is_training)

        branch5x5 = conv2d_factory(x, 32, (1, 1), is_training=is_training)
        branch5x5 = conv2d_factory(branch5x5, 16, (5, 5), is_training=is_training)

        branch3x3 = conv2d_factory(x, 32, (1, 1), is_training=is_training)
        branch3x3 = conv2d_factory(branch3x3, 32, (3, 3), is_training=is_training)
        branch3x3 = conv2d_factory(branch3x3, 32, (3, 3), is_training=is_training)

        branch_pool = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_factory(branch_pool, 32, (1, 1), is_training=is_training)

        x = concatenate([branch1x1, branch5x5, branch3x3, branch_pool], axis=-1)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # stage 2 16x16x128
    for _ in range(2):
        branch1x1 = conv2d_factory(x, 64, (1, 1), is_training=is_training)

        branch5x5 = conv2d_factory(x, 64, (1, 1), is_training=is_training)
        branch5x5 = conv2d_factory(branch5x5, 64, (5, 1), is_training=is_training)
        branch5x5 = conv2d_factory(branch5x5, 64, (1, 5), is_training=is_training)

        branch7x7 = conv2d_factory(x, 64, (1, 1), is_training=is_training)
        branch7x7 = conv2d_factory(branch7x7, 64, (7, 1), is_training=is_training)
        branch7x7 = conv2d_factory(branch7x7, 64, (1, 7), is_training=is_training)
        branch7x7 = conv2d_factory(branch7x7, 64, (7, 1), is_training=is_training)
        branch7x7 = conv2d_factory(branch7x7, 64, (1, 7), is_training=is_training)

        branch_pool = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_factory(branch_pool, 64, (1, 1), is_training=is_training)

        x = concatenate([branch1x1, branch5x5, branch7x7, branch_pool], axis=-1)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # stage 3 8x8x256
    for _ in range(2):
        branch1x1 = conv2d_factory(x, 128, (1, 1), is_training=is_training)

        branch5x5 = conv2d_factory(x, 128, (1, 1), is_training=is_training)
        branch5x5 = conv2d_factory(branch5x5, 128, (5, 1), is_training=is_training)
        branch5x5 = conv2d_factory(branch5x5, 128, (1, 5), is_training=is_training)

        branch3x3_2 = conv2d_factory(x, 128, (1, 1), is_training=is_training)
        branch3x3_2 = conv2d_factory(branch3x3_2, 128, (3, 3), is_training=is_training)
        branch3x3_2 = conv2d_factory(branch3x3_2, 128, (3, 1), is_training=is_training)
        branch3x3_2 = conv2d_factory(branch3x3_2, 128, (1, 3), is_training=is_training)

        branch_pool = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_factory(branch_pool, 128, (1, 1), is_training=is_training)

        x = concatenate([branch1x1, branch5x5, branch3x3_2, branch_pool], axis=-1)

    x = GlobalAveragePooling2D()(x)
    x = drop_connect_dense(x, nb_classes, activation='softmax',
                           kernel_regularizer=l2(weight_decay),
                           bias_regularizer=l2(weight_decay),
                          drop_prob=0.5,
                          is_training=is_training)

    model = Model(inputs=model_input, outputs=x)
    return model
