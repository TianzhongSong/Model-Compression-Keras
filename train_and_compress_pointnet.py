from utils.point_cloud_data_loader import DataGenerator
from models.pointnet import PointNet
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from copy import deepcopy
from compression import prune_weights, save_compressed_weights
import os
import argparse


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))
        fp.close()


def schedule(epoch):
    if epoch < 50:
        return 0.001
    else:
        return 0.0001


def main():
    nb_classes = 40
    train_file = './ModelNet40/ply_data_train.h5'
    test_file = './ModelNet40/ply_data_test.h5'

    epochs = 80
    fine_tune_epochs = 20
    batch_size = 32

    train = DataGenerator(train_file, batch_size, nb_classes, train=True)
    val = DataGenerator(test_file, batch_size, nb_classes, train=False)

    model = PointNet(nb_classes)
    model.summary()
    lr = 0.001
    adam = Adam(lr=lr)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    if not os.path.exists('./results/'):
        os.mkdir('./results/')
    learning_rate_scheduler = LearningRateScheduler(schedule=schedule)
    # pre-train
    history = model.fit_generator(train.generator(),
                                  steps_per_epoch=9840 // batch_size,
                                  epochs=epochs,
                                  validation_data=val.generator(),
                                  validation_steps=2468 // batch_size,
                                  callbacks=[learning_rate_scheduler],
                                  verbose=1)

    save_history(history, './results/')
    model.save_weights('./results/pointnet_weights.h5')

    # prune weights
    # save masks for weight layers
    masks = {}
    layer_count = 0
    # not compress first convolution layer
    first_conv = True
    for layer in model.layers:
        weight = layer.get_weights()
        if len(weight) >= 2:
            if not first_conv:
                w = deepcopy(weight)
                tmp, mask = prune_weights(w[0], compress_rate=args.compress_rate)
                masks[layer_count] = mask
                w[0] = tmp
                layer.set_weights(w)
            else:
                first_conv = False
        layer_count += 1
    # evaluate model after pruning
    model.compile(optimizer=Adam(lr=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    score = model.evaluate_generator(val.data_generator(), steps=2468 // batch_size)
    print('val loss: {}'.format(score[0]))
    print('val acc: {}'.format(score[1]))
    # fine-tune
    for i in range(fine_tune_epochs):
        for X, Y in train.data_generator():
            # train on each batch
            model.train_on_batch(X, Y)
            # apply masks
            for layer_id in masks:
                w = model.layers[layer_id].get_weights()
                w[0] = w[0] * masks[layer_id]
                model.layers[layer_id].set_weights(w)
        score = model.evaluate_generator(val.data_generator(), steps=2468 // batch_size)
        print('val loss: {}'.format(score[0]))
        print('val acc: {}'.format(score[1]))

    # save compressed weights
    compressed_name = './results/compressed_pointnet_weights'
    save_compressed_weights(model, compressed_name)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--compress-rate', type=float, default=0.9)
    args = parse.parse_args()
    main()
