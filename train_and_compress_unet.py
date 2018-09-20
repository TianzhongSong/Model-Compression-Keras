from models import unet
from utils import seg_data
from keras.optimizers import Adam, SGD
import os
from copy import deepcopy
from compression import prune_weights, save_compressed_weights
import argparse


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
        fp.close()


def main():
    nClasses = 2
    batch_size = 16
    epochs = 50
    fine_tune_epochs = 20
    img_height = 256
    img_width = 256
    root_path = '../../datasets/'
    mode = 'seg' if nClasses == 2 else 'parse'
    train_file = './data/{}_train.txt'.format(mode)
    val_file = './data/{}_test.txt'.format(mode)
    model = unet.Unet(nClasses, input_height=img_height, input_width=img_width)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-4),
                  metrics=['accuracy'])
    model.summary()

    train = seg_data.generator(root_path, train_file, batch_size, nClasses, img_height, img_width)

    val = seg_data.generator(root_path, val_file, batch_size, nClasses, img_height, img_width, train=False)

    if not os.path.exists('./results/'):
        os.mkdir('./results')
    history = model.fit_generator(train,
                                  steps_per_epoch=12706 // batch_size,
                                  validation_data=val,
                                  validation_steps=5000 // batch_size,
                                  epochs=epochs)
    save_history(history, './results/', 'unet')
    model.save_weights('./results/Unet_weights.h5')

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
    score = model.evaluate_generator(val, steps=5000 // batch_size)
    print('val loss: {}'.format(score[0]))
    print('val acc: {}'.format(score[1]))
    # fine-tune
    for i in range(fine_tune_epochs):
        for X, Y in seg_data.generator(root_path, train_file, batch_size, nClasses, img_height, img_width):
            # train on each batch
            model.train_on_batch(X, Y)
            # apply masks
            for layer_id in masks:
                w = model.layers[layer_id].get_weights()
                w[0] = w[0] * masks[layer_id]
                model.layers[layer_id].set_weights(w)
        score = model.evaluate_generator(seg_data.generator(root_path, val_file, batch_size, nClasses, img_height, img_width, train=False), steps=5000 // batch_size)
        print('val loss: {}'.format(score[0]))
        print('val acc: {}'.format(score[1]))

    # save compressed weights
    compressed_name = './results/compressed_unet_weights'
    save_compressed_weights(model, compressed_name)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--compress-rate', type=float, default=0.9)
    args = parse.parse_args()
    main()
