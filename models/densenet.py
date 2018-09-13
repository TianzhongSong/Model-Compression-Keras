from keras.layers import GlobalAveragePooling2D, concatenate, BatchNormalization, Activation
from keras.layers import Input, AveragePooling2D, Conv2D, Dense
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


def dense_block(x, growth_rate, nb_layers, nb_channels):
    x_list = [x]
    for i in range(nb_layers):
        feat = conv2d_factory(x, growth_rate)
        x_list.append(feat)
        x = concatenate(x_list, axis=-1)
        nb_channels += growth_rate
    return x, nb_channels


def transition_layer(x, nb_channels, reduce_rate):
    nb_channels = int(nb_channels * reduce_rate)
    x = conv2d_factory(x, nb_channels, kernel_size=(1, 1))
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    return x, nb_channels


def densenet(nb_classes, growth_rate, reduce_rate=0.5,
             depth=40, weight_decay=1E-4):
    assert (depth - 4) % 3 == 0, '(depth - 4) mod 3 must be 0'

    blocks = 3
    dense_layers = int((depth - 4) / 3)
    nb_channels = 2 * growth_rate
    model_input = Input(shape=(32, 32, 3))

    # init convolution
    x = conv2d_factory(model_input, nb_channels)

    for block in range(blocks - 1):
        x, nb_channels =  dense_block(x, growth_rate, dense_layers,
                                      nb_channels)
        x, nb_channels = transition_layer(x, nb_channels, reduce_rate)

    x, nb_channels = dense_block(x, growth_rate, dense_layers, nb_channels)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_classes, activation='softmax',
               kernel_regularizer=l2(weight_decay),
               bias_regularizer=l2(weight_decay))(x)

    model = Model(inputs=model_input, outputs=x)
    return model
