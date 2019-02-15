import json
import pandas as pd
import os


def json_to_csv(path):
    json_read = open(path).read()
    annotations = json.loads(json_read)
    annotations = list(annotations.values())
    annotations = [a for a in annotations if a['regions']]
    column_value = []
    for a in annotations:
        """When using ['region_attributes']"""
        '''
        point_num = [num['region_attributes'] for num in a['regions'].values()]
        point_xy = [xy['shape_attributes'] for xy in a['regions'].values()]

        point_x = [x['cx'] for x in point_xy]
        point_y = [y['cy'] for y in point_xy]

        # print(a['filename'])
        # print(point_xy)
        # print(point_num)

        # print(point_num[0]['point'], point_xy[0]['cx'], point_xy[0]['cy'])
        # print(point_num[1]['point'], point_xy[1]['cx'], point_xy[1]['cy'])
        # print(point_num[2]['point'], point_xy[2]['cx'], point_xy[2]['cy'])

        x1, y1, x2, y2, x3, y3 = 0, 0, 0, 0, 0, 0
        for i in range(len(point_num)):
            if point_num[i]['point'] == '1':
                x1 = point_xy[i]['cx']
                y1 = point_xy[i]['cy']
            elif point_num[i]['point'] == '2':
                x2 = point_xy[i]['cx']
                y2 = point_xy[i]['cy']
            elif point_num[i]['point'] == '3':
                x3 = point_xy[i]['cx']
                y3 = point_xy[i]['cy']
        '''

        data0_x = a['regions']['0']['shape_attributes']['cx']
        data0_y = a['regions']['0']['shape_attributes']['cy']
        data1_x = a['regions']['1']['shape_attributes']['cx']
        data1_y = a['regions']['1']['shape_attributes']['cy']
        data2_x = a['regions']['2']['shape_attributes']['cx']
        data2_y = a['regions']['2']['shape_attributes']['cy']
        data3_x = a['regions']['3']['shape_attributes']['cx']
        data3_y = a['regions']['3']['shape_attributes']['cy']
        data4_x = a['regions']['4']['shape_attributes']['cx']
        data4_y = a['regions']['4']['shape_attributes']['cy']
        data5_x = a['regions']['5']['shape_attributes']['cx']
        data5_y = a['regions']['5']['shape_attributes']['cy']

        column_value.append((data0_x, data0_y,
                             data1_x, data1_y,
                             data2_x, data2_y,
                             data3_x, data3_y,
                             data4_x, data4_y,
                             data5_x, data5_y,
                             a['filename']))
        # print("")

    return column_value


if __name__ == '__main__':

    data_path = './ankle_data'
    for data_name in ['train']:
        column_value = json_to_csv(path='{0}/{1}.json'.format(data_path, data_name))
        column_name = ['left_tiptoe_x', 'left_tiptoe_y',
                       'left_anklebone_x', 'left_anklebone_y',
                       'left_ankle_x', 'left_ankle_y',
                       'right_tiptoe_x', 'right_tiptoe_y',
                       'right_anklebone_x', 'right_anklebone_y',
                       'right_ankle_x', 'right_ankle_y',
                       'image']
        save_file = '{0}/{1}.csv'.format(data_path, data_name)
        if os.path.exists(save_file) == True:
            with open(save_file, 'w') as f:
                xml_df = pd.DataFrame(column_value, columns=column_name)
                xml_df.to_csv(f, header=True, index=False)
        else:
            xml_df = pd.DataFrame(columns=column_name)
            xml_df.to_csv(save_file, index=None)
            with open(save_file, 'w') as f:
                xml_df = pd.DataFrame(column_value, columns=column_name)
                xml_df.to_csv(f, header=True, index=False)

        print("Successfully converted json to csv.")
