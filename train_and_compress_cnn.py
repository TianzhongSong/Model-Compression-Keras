import keras
from utils.load_cifar import load_data
from keras.preprocessing.image import ImageDataGenerator
from models import resnet, densenet, inception, vggnet
from keras.callbacks import LearningRateScheduler
from compression import prune_weights, save_compressed_weights
from copy import deepcopy
import argparse
import os


def save_history(history, result_dir, prefix):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, '{}_result.txt'.format(prefix)), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))


def schedule(epoch):
    if epoch < 60:
        return 0.1
    elif epoch < 120:
        return 0.01
    elif epoch < 160:
        return 0.001
    else:
        return 0.0001


def training():
    batch_size = 64
    epochs = 200
    fine_tune_epochs = 30
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
    data_iter = datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True)

    if args.model == 'resnet':
        model = resnet.resnet(nb_classes,
                              depth=args.depth,
                              wide_factor=args.wide_factor)
        save_name = 'resnet_{}_{}_{}'.format(args.depth, args.wide_factor, args.data)
    elif args.model == 'densenet':
        model = densenet.densenet(nb_classes, args.growth_rate, depth=args.depth)
        save_name = 'densenet_{}_{}_{}'.format(args.depth, args.growth_rate, args.data)
    elif args.model == 'inception':
        model = inception.inception_v3(nb_classes)
        save_name = 'inception_{}'.format(args.data)

    elif args.model == 'vgg':
        model = vggnet.vgg(nb_classes)
        save_name = 'vgg_{}'.format(args.data)
    else:
        raise ValueError('Does not support {}'.format(args.model))

    model.summary()
    learning_rate_scheduler = LearningRateScheduler(schedule=schedule)
    if args.model == 'vgg':
        callbacks = None
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    else:
        callbacks = [learning_rate_scheduler]
        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=True),
                      metrics=['accuracy'])
    # pre-train
    history = model.fit_generator(data_iter,
                                  steps_per_epoch=x_train.shape[0] // batch_size,
                                  epochs=epochs,
                                  callbacks=callbacks,
                                  validation_data=(x_test, y_test))
    if not os.path.exists('./results/'):
        os.mkdir('./results/')
    save_history(history, './results/', save_name)
    model.save_weights('./results/{}_weights.h5'.format(save_name))

    # prune weights
    # save masks for weight layers
    masks = {}
    layer_count = 0
    for layer in model.layers:
        weight = layer.get_weights()
        if len(weight) >= 2:
            w = deepcopy(weight)
            tmp, mask = prune_weights(w[0], compress_rate=args.compress_rate)
            masks[layer_count] = mask
            w[0] = tmp
            layer.set_weights(w)
        layer_count += 1
    # evaluate model after pruning
    score = model.evaluate(x_test, y_test, verbose=0)
    print('val loss: {}'.format(score[0]))
    print('val acc: {}'.format(score[1]))
    # fine-tune
    for i in range(fine_tune_epochs):
        for _ in range(x_train.shape[0] // batch_size):
            X, Y = data_iter.next()
            # train on each batch
            model.train_on_batch(X, Y)
            # apply masks
            for layer_id in masks:
                w = model.layers[layer_id].get_weights()
                w[0] = w[0] * masks[layer_id]
                model.layers[layer_id].set_weights(w)
        score = model.evaluate(x_test, y_test, verbose=0)
        print('val loss: {}'.format(score[0]))
        print('val acc: {}'.format(score[1]))

    # save compressed weights
    compressed_name = './results/compressed_{}_weights'.format(args.model)
    save_compressed_weights(model, compressed_name)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--data', type=str, default='c10', help='Supports c10 (CIFAR-10) and c100 (CIFAR-100)')
    parse.add_argument('--model', type=str, default='resnet')
    parse.add_argument('--depth', type=int, default=50)
    parse.add_argument('--growth-rate', type=int, default=12, help='growth rate for densenet')
    parse.add_argument('--wide-factor', type=int, default=1, help='wide factor for WRN')
    parse.add_argument('--compress-rate', type=float, default=0.9)
    args = parse.parse_args()

    if args.data not in ['c10', 'c100']:
        raise Exception('args.data must be c10 or c100!')

    training()
