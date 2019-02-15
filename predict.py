import numpy as np
import cv2

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from keras.models import load_model


if __name__ == '__main__':

    """Load Training Data"""
    (x_train, x_test) = np.load('./ankle_data/images.npy')
    (y_train, y_test) = np.load('./ankle_data/points.npy')
    validation_rate = int(x_train.shape[0] * 0.2)
    x_train, y_train = x_train[:-validation_rate], y_train[:-validation_rate]
    x_test, y_test = x_test[-validation_rate:], y_test[-validation_rate:]

    """Load Saved Model"""
    model = load_model('./save_models/keyPointNet.h5')

    """Test Image"""
    # img_path = './test_img/test_0001.jpg'
    img_path = './ankle_data/images/ankle_00001.jpg'
    img = cv2.imread(img_path)

    """Data Normalization"""
    img = cv2.resize(img, (300, 300))
    test_img = img.astype('float32')
    test_img = test_img / 255.
    test_img = test_img[np.newaxis, :, :]

    output_pipe = make_pipeline(
        MinMaxScaler(feature_range=(-1, 1))
    )
    output_pipe.fit_transform(y_train)
    # output_pipe.fit_transform(y_test)

    """Prediction"""
    predictions = model.predict(test_img)
    xy_predictions = output_pipe.inverse_transform(predictions).reshape(6, 2)

    """Visualization"""
    circle_radian = 3
    circle_color_green = (0, 255, 0)
    circle_thickness = -1
    for i in xy_predictions:
        x, y = i[0], i[1]
        cv2.circle(img, (x, y), circle_radian, circle_color_green, circle_thickness)

    line_color_blue = (255, 0, 0)
    line_thickness = 1
    cv2.line(img, (xy_predictions[0][0], xy_predictions[0][1]), (xy_predictions[1][0], xy_predictions[1][1]), line_color_blue, line_thickness)
    cv2.line(img, (xy_predictions[1][0], xy_predictions[1][1]), (xy_predictions[2][0], xy_predictions[2][1]), line_color_blue, line_thickness)
    cv2.line(img, (xy_predictions[3][0], xy_predictions[3][1]), (xy_predictions[4][0], xy_predictions[4][1]), line_color_blue, line_thickness)
    cv2.line(img, (xy_predictions[4][0], xy_predictions[4][1]), (xy_predictions[5][0], xy_predictions[5][1]), line_color_blue, line_thickness)

    cv2.imshow('test', cv2.resize(img, (640, 480)))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
