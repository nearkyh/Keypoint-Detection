import numpy as np
import cv2
import argparse

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model


parser = argparse.ArgumentParser()
parser.add_argument('--video',
                    default=None,
                    type=str,
                    help="Input Video")
parser.add_argument('--camera',
                    default=0,
                    type=int,
                    help="Camera Index")
args = parser.parse_args()

input_video = None
if args.video == None:
    input_video = args.camera
else:
    input_video = args.video


if __name__ == '__main__':

    """Load Training Data"""
    (x_train, x_test) = np.load('./ankle_data/images.npy')
    (y_train, y_test) = np.load('./ankle_data/points.npy')
    validation_rate = int(x_train.shape[0] * 0.2)
    x_train, y_train = x_train[:-validation_rate], y_train[:-validation_rate]
    x_test, y_test = x_test[-validation_rate:], y_test[-validation_rate:]

    """Data Normalization"""
    output_pipe = make_pipeline(
        MinMaxScaler(feature_range=(-1, 1))
    )
    output_pipe.fit_transform(y_train)
    # output_pipe.fit_transform(y_test)

    """Load Saved Model"""
    model = load_model('./save_models/keyPointNet.h5')

    """Input Video"""
    cap = cv2.VideoCapture(input_video)
    if cap.isOpened() == False:
        print('ERROR, Can\'t open the VIDEO({0})'.format(str(input_video)))
        exit()

    while(cap.isOpened()):
        ret, image_np = cap.read()

        """Data Normalization"""
        image_np = cv2.resize(image_np, (300, 300))
        input_img = image_np.astype('float32')
        input_img = input_img / 255.
        input_img = input_img[np.newaxis, :, :]

        """Prediction"""
        predictions = model.predict(input_img)
        xy_predictions = output_pipe.inverse_transform(predictions).reshape(6, 2)

        predict_top = model.predict_classes(input_img)
        predict_top = predict_top[0]
        predict_acc= predictions[0][predict_top]
        predict_acc = float("{0:.2f}".format(predict_acc*100))
        print("Accuracy {0}%".format(predict_acc))

        min_score_thresh = 0
        if predict_acc >= min_score_thresh:
            """Visualization"""
            circle_radian = 3
            circle_color_green = (0, 255, 0)
            circle_thickness = -1
            for i in xy_predictions:
                x, y = i[0], i[1]
                cv2.circle(image_np, (x, y), circle_radian, circle_color_green, circle_thickness)

            line_color_blue = (255, 0, 0)
            line_thickness = 1
            cv2.line(image_np, (xy_predictions[0][0], xy_predictions[0][1]), (xy_predictions[1][0], xy_predictions[1][1]), line_color_blue, line_thickness)
            cv2.line(image_np, (xy_predictions[1][0], xy_predictions[1][1]), (xy_predictions[2][0], xy_predictions[2][1]), line_color_blue, line_thickness)
            cv2.line(image_np, (xy_predictions[3][0], xy_predictions[3][1]), (xy_predictions[4][0], xy_predictions[4][1]), line_color_blue, line_thickness)
            cv2.line(image_np, (xy_predictions[4][0], xy_predictions[4][1]), (xy_predictions[5][0], xy_predictions[5][1]), line_color_blue, line_thickness)

        cv2.imshow('Testing', cv2.resize(image_np, (640, 480)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()
