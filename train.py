import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import TensorBoard, EarlyStopping
from keras.utils.training_utils import multi_gpu_model

from keypoint_net import KeyPointNet
from mobilenet_v2 import MobileNetv2


def history_visualization(model_info):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(range(1, len(model_info.history['acc']) + 1), model_info.history['acc'])
    axs[0].plot(range(1, len(model_info.history['val_acc']) + 1), model_info.history['val_acc'])
    axs[0].set_title('model_info Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_info.history['acc']) + 1), len(model_info.history['acc']) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1, len(model_info.history['loss']) + 1), model_info.history['loss'])
    axs[1].plot(range(1, len(model_info.history['val_loss']) + 1), model_info.history['val_loss'])
    axs[1].set_title('model_info Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_info.history['loss']) + 1), len(model_info.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()


if __name__ == '__main__':

    """Load Training Data"""
    (x_train, x_test) = np.load('./ankle_data/images.npy')
    (y_train, y_test) = np.load('./ankle_data/points.npy')
    validation_rate = int(x_train.shape[0] * 0.2)
    x_train, y_train = x_train[:-validation_rate], y_train[:-validation_rate]
    x_test, y_test = x_test[-validation_rate:], y_test[-validation_rate:]

    """Image Normalization"""
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255.
    x_test /= 255.

    """"Key Point Normalization"""
    output_pipe = make_pipeline(
        MinMaxScaler(feature_range=(-1, 1))
    )
    y_train = output_pipe.fit_transform(y_train)
    y_test = output_pipe.fit_transform(y_test)

    """Define Model"""
    pointNet = KeyPointNet(input_shape=x_train.shape[1:],
                           num_class=12)
    model = pointNet.build()
    # model = MobileNetv2(input_shape=x_train.shape[1:],
    #                     k=12)

    """Define Optimizer"""
    opt = Adam()
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.95, nesterov=True)
    rms = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    """Define Callback"""
    earlyStop = EarlyStopping(monitor='val_acc', patience=30, verbose=0, mode='auto')
    tensorBoard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

    """Training"""
    # model = multi_gpu_model(model, gpus=2)
    model.compile(loss='mse', optimizer=rms, metrics=['accuracy'])
    model_info = model.fit(x_train, y_train,
                           batch_size=24,
                           epochs=50,
                           validation_data=(x_test, y_test),
                           verbose=1,
                           shuffle=True,
                           callbacks=[tensorBoard, earlyStop])
    model.summary()
    score = model.evaluate(x_test, y_test, verbose=0)

    """Result"""
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    history_visualization(model_info=model_info)

    """Save Weight"""
    try:
        if not(os.path.isdir('./save_models')):
            os.makedirs(os.path.join('./save_models'))
    except Exception as e:
        print("Failed to create directory!!!")
    model.save('./save_models/keyPointNet.h5')
