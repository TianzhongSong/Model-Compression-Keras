import keras
from utils.load_cifar import load_data
from keras.preprocessing.image import ImageDataGenerator
from models.inception_v3 import inception_v3
from utils import compression
from utils.schedules import onetenth_120_160
import argparse
import os


def save_history(history, result_dir, prefix):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result_{}.txt'.format(prefix)), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))


def training():
    batch_size = 64
    epochs = 200
    lr = 0.1

    x_train, y_train, x_test, y_test, nb_classes = load_data(args.data)
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    y_train = keras.utils.to_categorical(y_train, nb_classes)
    y_test = keras.utils.to_categorical(y_test, nb_classes)

    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=5. / 32,
                                 height_shift_range=5. / 32)
    data_iter = datagen.flow(x_train, y_train, batch_size=args.batch_size, shuffle=True)

    model = inception_v3(nb_classes, is_training=True)

    opt = keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    history = model.fit_generator(data_iter,
                                  steps_per_epoch=x_train.shape[0] // batch_size,
                                  epochs=epochs,
                                  callbacks=[onetenth_120_160(lr)],
                                  validation_data=(x_test, y_test))
    if not os.path.exists('./results/'):
        os.mkdir('./results/')
    save_history(history, './results/', 'inception')
    model.save_weights('./results/inception_weights.h5')

    dropped_weights = compression.select_best_model(model, x_test, y_test, iter=100)
    compression.save_pruned_weights(dropped_weights, 'inception_compressed_weights')


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--data', type=str, default='c10', help='Supports c10 (CIFAR-10) and c100 (CIFAR-100)')
    args = parse.parse_args()

    if args.data not in ['c10', 'c100']:
        raise Exception('args.data must be c10 or c100!')

    training()
