from keras.layers import Flatten, MaxPooling2D, Input
from utils.keras_wrapped_layers import drop_connect_conv_2d, drop_connect_dense
from keras.models import Model


def cnn(nb_classes, is_training=True):
    model_input = Input(shape=(32, 32, 3))

    x = drop_connect_conv_2d(model_input, 32, (3, 3), padding='same', activation='relu',
                             drop_prob=0.0, is_training=is_training)
    x = drop_connect_conv_2d(x, 32, (3, 3), padding='same', activation='relu',
                             drop_prob=0.5, is_training=is_training)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = drop_connect_conv_2d(x, 64, (3, 3), padding='same', activation='relu',
                             drop_prob=0.5, is_training=is_training)
    x = drop_connect_conv_2d(x, 64, (3, 3), padding='same', activation='relu',
                             drop_prob=0.5, is_training=is_training)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = drop_connect_conv_2d(x, 128, (3, 3), padding='same', activation='relu',
                             drop_prob=0.5, is_training=is_training)
    x = drop_connect_conv_2d(x, 128, (3, 3), padding='same', activation='relu',
                             drop_prob=0.5, is_training=is_training)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = Flatten()(x)

    x = drop_connect_dense(x, 512, activation='relu', drop_prob=0.5, is_training=is_training)
    x = drop_connect_dense(x, nb_classes, activation='softmax', drop_prob=0.5, is_training=is_training)

    model = Model(inputs=model_input, outputs=x)

    return model
