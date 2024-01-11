from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow import keras


from sklearn.metrics import classification_report
import numpy as np


def EEGNetTF(nb_classes, Chans = 64, Samples = 128,
             dropoutRate = 0.5, kernLength = 128, F1 = 16,
             D = 2, F2 = 32, norm_rate = 0.25, dropoutType = 'Dropout'):

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout

    input1 = Input(shape = (Chans, Samples, 1))

    block1 = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias = False,
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name = 'flatten')(block2)

    dense = Dense(nb_classes, name = 'dense',
                  kernel_constraint = max_norm(norm_rate), activation='elu')(flatten)

    softmax = Activation('softmax', name = 'softmax')(dense)

    return Model(inputs=input1, outputs=softmax)


def train_cnn_tf(x_train, x_test, y_train, y_test, enable_logging=False):
    x_train, x_test, y_train, y_test = x_train.copy(), x_test.copy(), y_train.copy(), y_test.copy()
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    model = EEGNetTF(nb_classes = 4, Chans = x_train.shape[1], Samples = x_train.shape[2],
                     dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16,
                     dropoutType = 'Dropout')

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=2e-4, weight_decay=1e-2), #
                  metrics = ['accuracy'])
    model.fit(x_train, y_train, epochs=500, batch_size=64, validation_data=(x_test, y_test))

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print('Точность на тестовых данных:', test_accuracy)

    model.save('./saved_model/')

    probs = model.predict(x_test)
    preds = probs.argmax(axis = -1)
    acc = np.mean(preds == y_test)

    print(classification_report(y_test, preds))

    return acc
