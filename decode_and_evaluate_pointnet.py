from models import pointnet
from compression import decode_weights
from utils.point_cloud_data_loader import DataGenerator


if __name__ == '__main__':
    nb_classes = 40
    model = pointnet.PointNet(nb_classes)
    model.summary()
    test_file = './ModelNet40/ply_data_test.h5'
    val = DataGenerator(test_file, 32, nb_classes, train=False)

    weights = decode_weights('./results/compressed_pointnet_weights.h5')
    for i in range(len(model.layers)):
        if model.layers[i].name in weights:
            weight = [w for w in weights[model.layers[i].name]]
            model.layers[i].set_weights(weight)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    score = model.evaluate_generator(val.data_generator(), steps=2468 // 32)
    print('val loss: {}'.format(score[0]))
    print('val acc: {}'.format(score[1]))
