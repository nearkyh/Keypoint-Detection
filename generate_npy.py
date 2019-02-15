import pandas as pd
import os
import cv2
import numpy as np


def fixed_image_resize():
    height = 300
    width = 300

    return height, width


def fixed_point_resize(img, x_point, y_point):
    img_h, img_w = img.shape[0], img.shape[1]
    fixed_img_h, fixed_img_w = fixed_image_resize()
    y = int(fixed_img_h / img_h * y_point)
    x = int(fixed_img_w / img_w * x_point)

    return x, y


def create_data(csv_path='./data/train.csv',
                image_path='./images/train'):
    input_shape = fixed_image_resize()

    train_df = pd.read_csv(csv_path)
    data0_x = train_df['left_tiptoe_x'].tolist()
    data0_y = train_df['left_tiptoe_y'].tolist()
    data1_x = train_df['left_anklebone_x'].tolist()
    data1_y = train_df['left_anklebone_y'].tolist()
    data2_x = train_df['left_ankle_x'].tolist()
    data2_y = train_df['left_ankle_y'].tolist()
    data3_x = train_df['right_tiptoe_x'].tolist()
    data3_y = train_df['right_tiptoe_y'].tolist()
    data4_x = train_df['right_anklebone_x'].tolist()
    data4_y = train_df['right_anklebone_y'].tolist()
    data5_x = train_df['right_ankle_x'].tolist()
    data5_y = train_df['right_ankle_y'].tolist()
    image = train_df['image'].tolist()

    x_data = []
    for i in image:
        train_img = cv2.imread('{0}/{1}'.format(image_path, i))
        train_img = cv2.resize(train_img, input_shape)
        x_data.append(train_img)

    y_data = []
    for i in range(len(data0_x)):
        data0_x[i], data0_y[i] = fixed_point_resize(img=cv2.imread('{0}/{1}'.format(image_path, image[0])),
                                                    x_point=data0_x[i],
                                                    y_point=data0_y[i])
        data1_x[i], data1_y[i] = fixed_point_resize(img=cv2.imread('{0}/{1}'.format(image_path, image[0])),
                                                    x_point=data1_x[i],
                                                    y_point=data1_y[i])
        data2_x[i], data2_y[i] = fixed_point_resize(img=cv2.imread('{0}/{1}'.format(image_path, image[0])),
                                                    x_point=data2_x[i],
                                                    y_point=data2_y[i])
        data3_x[i], data3_y[i] = fixed_point_resize(img=cv2.imread('{0}/{1}'.format(image_path, image[0])),
                                                    x_point=data3_x[i],
                                                    y_point=data3_y[i])
        data4_x[i], data4_y[i] = fixed_point_resize(img=cv2.imread('{0}/{1}'.format(image_path, image[0])),
                                                    x_point=data4_x[i],
                                                    y_point=data4_y[i])
        data5_x[i], data5_y[i] = fixed_point_resize(img=cv2.imread('{0}/{1}'.format(image_path, image[0])),
                                                    x_point=data5_x[i],
                                                    y_point=data5_y[i])
        y_data.append([data0_x[i], data0_y[i],
                       data1_x[i], data1_y[i],
                       data2_x[i], data2_y[i],
                       data3_x[i], data3_y[i],
                       data4_x[i], data4_y[i],
                       data5_x[i], data5_y[i]])

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    print('Successfully created the data')
    
    return x_data, y_data


def generate_npy(x_train, y_train, x_test, y_test,
                 images_save_path='./data/images.npy',
                 points_save_path='./data/points.npy'):
    np.save(images_save_path, (x_train, x_test))
    np.save(points_save_path, (y_train, y_test))

    if os.path.exists(images_save_path):
        print("Successfully created the images.npy")
    if os.path.exists(points_save_path):
        print("Successfully created the points.npy")


if __name__ == '__main__':

    print('Generating ...')

    x_train, y_train = create_data(csv_path='./ankle_data/train.csv',
                                   image_path='./ankle_data/images')
    generate_npy(x_train=x_train,
                 y_train=y_train,
                 x_test=x_train,
                 y_test=y_train,
                 images_save_path='./ankle_data/images.npy',
                 points_save_path='./ankle_data/points.npy')
    # print(x_train.shape, y_train.shape)
