from models import unet
from compression import decode_weights
from utils.seg_data import generator


if __name__ == '__main__':
    nClasses = 2
    img_height = 256
    img_width = 256
    root_path = '../../datasets/'
    val_file = './data/seg_test.txt'
    model = unet.Unet(nClasses, input_height=img_height, input_width=img_width)
    model.summary()

    weights = decode_weights('./results/compressed_unet_weights.h5')
    for i in range(len(model.layers)):
        if model.layers[i].name in weights:
            weight = [w for w in weights[model.layers[i].name]]
            model.layers[i].set_weights(weight)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    val = generator(root_path, val_file, 16, nClasses, img_height, img_width, train=False)
    score = model.evaluate_generator(val, steps=5000 // 16)
    print('val loss: {}'.format(score[0]))
    print('val acc: {}'.format(score[1]))
