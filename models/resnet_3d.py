from keras.layers import Dense, Input, MaxPooling3D, Flatten, Activation
from keras.layers import BatchNormalization, add, AveragePooling3D, GlobalAveragePooling3D
from keras.models import Model
from utils.keras_wrapped_layers import drop_connect_dense, drop_connect_conv_3d


def conv_factory(x, nb_filters,
                 kernel_size=(3, 3, 3),
                 strides=(1, 1, 1),
                 padding='same',
                 is_training=True):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = drop_connect_conv_3d(x, nb_filters,
                             kernel_size,
                             strides=strides,
                             padding=padding,
                             use_bias=False,
                             drop_prob=0.5,
                             is_training=is_training)
    return x


def residual_block(x, nb_filters,
                   kernel_size=(3, 3, 3),
                   strides=(1, 1, 1),
                   padding='same',
                   is_training=True):
    x = conv_factory(x, nb_filters, kernel_size,
                     strides=strides, padding=padding,
                     is_training=is_training)
    x = conv_factory(x, nb_filters, (3, 3, 3), is_training=is_training)
    return x


def ResNet_3d(nb_classes, input_shape=(112, 112, 8, 3), is_training=True):
    nb_blocks = [2, 2, 2, 2]
    model_inputs = Input(shape=input_shape)

    # initial convolution
    x = drop_connect_conv_3d(model_inputs, 64, (3, 3, 3), padding='same',
                             drop_prob=0.0, is_training=is_training)

    # stage 1 56x56x8
    y = conv_factory(x, 64, (1, 1, 1), strides=(2, 2, 1),
                     padding='valid', is_training=is_training)

    x = residual_block(x, 64, (1, 1, 1),
                       strides=(2, 2, 1),
                       padding='valid',
                       is_training=is_training)

    y = add([x, y])
    for _ in range(nb_blocks[0] - 1):
        x = residual_block(y, 64, (3, 3, 3))
        y = add([x, y])

    # stage 2 28x28x4
    x = residual_block(y, 128, (1, 1, 1),
                       strides=(2, 2, 2),
                       padding='valid',
                       is_training=is_training)

    y = conv_factory(y, 128, (1, 1, 1), strides=(2, 2, 2),
                     padding='valid', is_training=is_training)

    y = add([x, y])
    for _ in range(nb_blocks[1] - 1):
        x = residual_block(y, 128, (3, 3, 3))
        y = add([x, y])

    # stage 3 14x14x2
    x = residual_block(y, 256, (1, 1, 1),
                       strides=(2, 2, 2),
                       padding='valid',
                       is_training=is_training)

    y = conv_factory(y, 256, (1, 1, 1), strides=(2, 2, 2),
                     padding='valid', is_training=is_training)

    y = add([x, y])
    for _ in range(nb_blocks[2] - 1):
        x = residual_block(y, 256, (3, 3, 3))
        y = add([x, y])

    # stage 4 7x7x1
    x = residual_block(y, 256, (1, 1, 1),
                       strides=(2, 2, 2),
                       padding='valid',
                       is_training=is_training)

    y = conv_factory(y, 256, (1, 1, 1), strides=(2, 2, 2),
                     padding='valid', is_training=is_training)

    y = add([x, y])
    for _ in range(nb_blocks[3] - 1):
        x = residual_block(y, 256, (3, 3, 3))
        y = add([x, y])

    x = BatchNormalization()(y)
    x = Activation('relu')(x)
    x = GlobalAveragePooling3D()(x)

    x = drop_connect_dense(x, nb_classes,
                           activation='softmax',
                           drop_prob=0.5,
                           is_training=is_training)

    model = Model(inputs=model_inputs, outputs=x)
    return model
