import keras
from models import resnet, densenet, inception, vggnet
from utils.load_cifar import load_data
from compression import decode_weights
import argparse


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--data', type=str, default='c10', help='Supports c10 (CIFAR-10) and c100 (CIFAR-100)')
    parse.add_argument('--model', type=str, default='resnet')
    parse.add_argument('--depth', type=int, default=50)
    parse.add_argument('--growth-rate', type=int, default=12, help='growth rate for densenet')
    parse.add_argument('--wide-factor', type=int, default=1, help='wide factor for WRN')
    args = parse.parse_args()

    x_train, y_train, x_test, y_test, nb_classes = load_data(args.data)
    y_test = keras.utils.to_categorical(y_test, nb_classes)

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

    weight_file = './results/' + save_name + '.h5'
    # decode
    weights = decode_weights(weight_file)
    for i in range(len(model.layers)):
        if model.layers[i].name in weights:
            weight = [w for w in weights[model.layers[i].name]]
            model.layers[i].set_weights(weight)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    score = model.evaluate(x_test, y_test, verbose=0)
    print('val loss: {}'.format(score[0]))
    print('val acc: {}'.format(score[1]))
