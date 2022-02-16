#!/usr/bin/env python3
"""
    Transfer Knowledge
"""
import tensorflow.keras as K
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.applications.inception_v3 import InceptionV3
from keras.utils.np_utils import to_categorical

from keras.layers import Lambda, Input

# to fix issue with saving keras applications
K.learning_phase = K.backend.learning_phase

# https://www.tensorflow.org/tutorials/load_data/images

def preprocess_data(X, Y):
    """
    Method:
        pre-processes the data for your model.

    Args:
        - X(numpy.ndarray), shape (m, 32, 32, 3)
            containing the CIFAR 10 data, where m is
            the number of data points.
        - Y(numpy.ndarray), shape (m,)
            containing the CIFAR 10 labels for X.

     Return:
        - X_p(numpy.ndarray): containing the preprocessed X
        - Y_p(numpy.ndarray): containing the preprocessed Y
    """
    # each Keras Application expects a specific kind of input preprocessing.
    # preprocesses a tensor or numpy array encoding a batch of images.
    # scale input pixels between -1 and 1
    X_p = K.applications.inception_v3.preprocess_input(X)

    # https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
    Y_p = K.utils.to_categorical(Y)

    return X_p, Y_p

#------------ Main -----------------#

# https://keras.io/guides/transfer_learning/ !!!

# https://rescale.com/neural-networks-using-keras-on-rescale/ !!!!!!! example of ciraf10


if __name__ == '__main__':

    ### ** Load data **

    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    # x_train: uint8 NumPy array of grayscale image data with shapes (50000, 32, 32, 3),
    # containing the training data. Pixel values range from 0 to 255.
    # y_train: uint8 NumPy array of labels (integers in range 0-9)
    # with shape (50000, 1) for the training data.
    # for now no testing data import
    assert x_train.shape == (50000, 32, 32, 3)
    assert y_train.shape == (50000, 1)

    ### ** Preprocess the data **

    X_prep, Y_prep = preprocess_data(x_train, y_train)
    X_prep_t, Y_prep_t = preprocess_data(x_test, y_test)

    ####   Model ######


    # https://stackoverflow.com/questions/42260265/resizing-an-input-image-in-a-keras-lambda-layer

    input_layer = Input(shape=(32, 32, 3))

    # Your first layer should be a lambda layer that scales up the data to the correct size
    # https://machinelearningmastery.com/a-gentle-introduction-to-channels-first-and-channels-last-image-formats-for-deep-learning/
    # Channels Last: [rows][cols][channels].
    # Channels First: [channels][rows][cols].

    resizing_layer = Lambda(lambda image: K.backend.resize_images(image, 299, 299, data_format="channels_last"))(input_layer)

    # Get the InceptionV3 model so we can do transfer learning
    base_inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

    # only if we want to freeze layers
    base_inception.trainable = False

    # Add a global spatial average pooling layer
    out = base_inception(resizing_layer)
    out = K.layers.Flatten()(out)
    # out = GlobalAveragePooling2D()(out)
    # out = Dense(512, activation='relu')(out)
    # out = Dense(512, activation='relu')(out)
    total_classes = Y_prep.shape[1]
    predictions = Dense(total_classes, activation='softmax')(out)

    model = Model(inputs=base_inception.input, outputs=predictions)


    # Compile
    model.compile(K.optimizers.Adam(lr=.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    # train the model on the new data for a few epochs
    model.fit(
        X_prep,
        Y_prep,
        batch_size=64,
        epochs=5)

    # model.save('cifar10.h5')
    print(X_prep_t.shape)
    model.evaluate(x_test, y_test, batch_size=128, verbose=1)
