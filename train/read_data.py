import pandas as pd
from pathlib import Path
import spatial_transition as st


def read_inputs(data_type, screen):

    if screen == "Screen1":
        screen_type = 1
    elif screen == "Screen2":
        screen_type = 2
    else:
        print("SOMETHING WRONG HAPPENS")
        return

    inputs = pd.DataFrame()

    # read input yolo data
    for label_file in Path("../dataset/arms/labels/" + data_type).iterdir():
    # for label_file in Path("../dataset/arms/yolov7-w6-test").iterdir():
        # read targets
        df = pd.read_csv(label_file, sep=" ", names=['id', 'center-x', 'center-y', 'w', 'h'])

        image_type = str(label_file.name).split('_')

        df['picture'] = [int(image_type[0])] * len(df.index)

        if image_type[1] == screen:
            df['screen'] = [screen_type] * len(df.index)
            inputs = inputs.append(df, ignore_index=True)

    inputs = inputs.sort_values(by=["picture", "id", "screen"]).reset_index(drop=True)
    return inputs


def read_outputs():

    df_train = pd.read_csv("../dataset/arms/targets/train_gt_coordinates.csv")
    df_val = pd.read_csv("../dataset/arms/targets/val_gt_coordinates.csv")
    df_test = pd.read_csv("../dataset/arms/targets/test_gt_coordinates.csv")

    df_train.columns = ["picture", "0.x", "0.z", "1.x", "1.z", "2.x", "2.z", "3.x", "3.z"]
    df_val.columns = ["picture", "0.x", "0.z", "1.x", "1.z", "2.x", "2.z", "3.x", "3.z"]
    df_test.columns = ["picture", "0.x", "0.z", "1.x", "1.z", "2.x", "2.z", "3.x", "3.z"]

    train_outputs = reorganize_outputs(df_train)
    val_outputs = reorganize_outputs(df_val)
    test_outputs = reorganize_outputs(df_test)

    return train_outputs, val_outputs, test_outputs


def reorganize_outputs(df):

    outputs_0 = df.loc[:, ['picture', '0.x', '0.z']]
    outputs_0['id'] = [0] * len(outputs_0.index)
    outputs_0.columns = ['picture', 'x', 'z', 'id']

    outputs_1 = df.loc[:, ['picture', '1.x', '1.z']]
    outputs_1['id'] = [1] * len(outputs_1.index)
    outputs_1.columns = ['picture', 'x', 'z', 'id']

    outputs_2 = df.loc[:, ['picture', '2.x', '2.z']]
    outputs_2['id'] = [2] * len(outputs_2.index)
    outputs_2.columns = ['picture', 'x', 'z', 'id']

    outputs_3 = df.loc[:, ['picture', '3.x', '3.z']]
    outputs_3['id'] = [3] * len(outputs_3.index)
    outputs_3.columns = ['picture', 'x', 'z', 'id']

    outputs = outputs_0.append(outputs_1, ignore_index=True).append(outputs_2, ignore_index=True).\
        append(outputs_3, ignore_index=True).\
        sort_values(by=['picture', 'id']).reset_index(drop=True)

    return outputs


def process_inputs(inputs, x1, x2, input_camera):
    """
    data format is: [xr1, zr1, w1, h1, xr3, zr3, w3, h3]
    if the one of the camera data is missing, it is set as 0
    """

    input_data = []

    for i in range(x1, x2 + 1):
        for j in range(0, 4):
            temp = inputs.loc[(inputs['picture'] == i) & (inputs['id'] == j)].reset_index(drop=True)

            # if there is only one or zero camera displays the robots, this data will be ignored
            # the data format is [0]
            if len(temp.index) == 0:
                temp1 = [0]
                # temp1 = [0, 0, 0, 0, 0, 0, 0, 0]
            elif len(temp.index) == 1:
                robot_point = st.s2r((temp.iloc[0]['center-x'], temp.iloc[0]['center-y']), input_camera)
                xr, zr = robot_point
                temp1 = [xr/808, zr/448, float(temp['w']), float(temp['h'])]

                # if temp.iloc[0]['screen'] == 1:
                    # temp1 = [0]
                    # temp1 = [float(temp['center-x']), float(temp['center-y']), float(temp['w']),
                    #          float(temp['h']), 0, 0, 0, 0]
                # else:
                    # temp1 = [0]
                    # temp1 = [0, 0, 0, 0, float(temp['center-x']), float(temp['center-y']),
                             # float(temp['w']), float(temp['h'])]

            # if the robots exist in two screens,
            # the data format is: [xr1, zr1, w1, h1, xr2, zr2, w2, h2]
            # elif len(temp.index) == 2:
            #     # xi, zi = get_intersect(robot_point1, robot_point3)
            #     # temp1 = [xi/808, zi/448]
            #     temp.sort_values(by=["screen"]).reset_index(drop=True)
            #     robot_point1 = st.s2r((temp.iloc[0]['center-x'], temp.iloc[0]['center-y']), "o1")
            #     robot_point3 = st.s2r((temp.iloc[1]['center-x'], temp.iloc[1]['center-y']), "o3")
            #     xr1, zr1 = robot_point1
            #     xr3, zr3 = robot_point3
            #     temp1 = [xr1/808, zr1/448,
            #              temp.iloc[0]['w'], temp.iloc[0]['h'],
            #              xr3/808, zr3/448,
            #              temp.iloc[1]['w'], temp.iloc[1]['h']]

            else:
                print(len(temp.index))
                print(temp)
                print("SOMETHING WRONG HAPPENS!")
                return

            input_data.append(temp1)

    return input_data


def process_outputs(outputs):
    output_data = []

    for index, row in outputs.iterrows():
        output_data.append([row['x']/808, row['z']/448])

    return output_data


def filter_data(inputs, outputs):
    """
    filter out the output data that does not exist in input data
    """
    input_data_1 = []
    output_data_1 = []

    for i in range(len(inputs)):
        if inputs[i] != [0]:
            input_data_1.append(inputs[i])
            output_data_1.append(outputs[i])

    print("Length of input data is: ", len(input_data_1))
    # print(input_data_1)

    print("Length of output data is: ", len(output_data_1))
    # print(output_data_1)

    return input_data_1, output_data_1


def get_intersect(point1, point3):
    (x1, y1, z1) = st.o1_point
    (x2, z2) = point1
    (x3, y3, z3) = st.o3_point
    (x4, z4) = point3

    k1 = (z1 - z2) / (x1 - x2)
    k2 = (z3 - z4) / (x3 - x4)
    x = (z2 - z4 + k2 * x4 - k1 * x2) / (k2 - k1)
    z = (x-x2)/(x1-x2)*(z1-z2) + z2
    return x, z
