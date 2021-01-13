from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv1D, Conv2D, PReLU, MaxPooling1D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
import numpy as np

np.random.seed(1) # set seed to guarantee reproducibility (ONLY IF TRAINED ON CPU!)


def get_model(name, input_shape, lr=0.001, alpha=0.001):
    if name == 'conv_model':
        return get_conv_model(input_shape, lr, alpha)
    elif name == 'conv_2d_model':
        return get_conv_2d_model(input_shape, lr, alpha)
    elif name == 'small_model':
        return get_small_model(input_shape, lr, alpha)
    elif name == 'perceptron':
        return get_perceptron(input_shape, lr, alpha)
    else:
        print('Unknown Model Name')


def get_conv_2d_model(input_shape, lr=0.001, alpha=0.001):
    model = Sequential()
    model.add(Conv2D(16, (2, 2), kernel_regularizer=l2(alpha), input_shape=input_shape))
    model.add(PReLU())
    model.add(Conv2D(16, (2, 2), kernel_regularizer=l2(alpha)))
    model.add(PReLU())
    model.add(Batchnormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), kernel_regularizer=l2(alpha)))
    model.add(PReLU())
    model.add(Conv2D(32, (3, 3), kernel_regularizer=l2(alpha)))
    model.add(PReLU())
    model.add(Batchnormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(20, kernel_regularizer=l2(alpha)))
    model.add(PReLU())
    model.add(Dropout(0.3))
    model.add(Dense(1, kernel_regularizer=l2(alpha), activation='sigmoid'))
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])

    # print model architecture
    # model.summary()
    return model


def get_conv_model(input_shape, lr=0.001, alpha=0.001):
    model = Sequential()
    model.add(Conv1D(16, 2, kernel_regularizer=l2(alpha), input_shape=input_shape))
    model.add(PReLU())
    model.add(Conv1D(16, 2, kernel_regularizer=l2(alpha)))
    model.add(PReLU())
    model.add(Batchnormalization())
    model.add(MaxPooling1D(2))
    model.add(Conv1D(32, 3, kernel_regularizer=l2(alpha)))
    model.add(PReLU())
    model.add(Conv1D(32, 3, kernel_regularizer=l2(alpha)))
    model.add(PReLU())
    model.add(Batchnormalization())
    model.add(MaxPooling1D(2))
    model.add(Dense(20, kernel_regularizer=l2(alpha)))
    model.add(PReLU())
    model.add(Dropout(0.3))
    model.add(Dense(1, kernel_regularizer=l2(alpha), activation='sigmoid'))
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])

    # print model architecture
    # model.summary()
    return model


def get_small_model(input_shape, lr=0.001, alpha=0.001):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(20, kernel_regularizer=l2(alpha)))
    model.add(PReLU())
    model.add(Dense(1, kernel_regularizer=l2(alpha), activation='sigmoid'))
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])

    # print model architecture
    # model.summary()
    return model


def get_perceptron(input_shape, lr=0.001, alpha=0.001):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(1, kernel_regularizer=l2(alpha), activation='sigmoid'))
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])

    # print model architecture
    # model.summary()
    return model


# convert feature list to needed shape with one channel at last position
def convert_features(features):
    features = np.array(features)
    features = features[:, :, None]
    return features


def train_model(features, labels, name='small_model', epochs=80, batch_size=20, validation_data=None, lr=0.001,
                alpha=0.001, file_path=None, class_weight=None):
    input_shape = (len(features[0]), 1)
    model = get_model(name, input_shape, lr, alpha)
    features = convert_features(features)
    model.fit(features, labels, batch_size=batch_size, epochs=epochs, validation_data=validation_data, verbose=0,
              class_weight=class_weight)
    if file_path:
        model.save(file_path)
    return model


def eval_model(model, features, labels):
    features = convert_features(features)
    return model.evaluate(features, labels, verbose=0)


def predict_model(model, features):
    features = convert_features(features)
    return model.predict(features)



