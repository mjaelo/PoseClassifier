import json

import xlrd
import pandas as pd

import sklearn.feature_extraction.image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

expected = []
poses_total = 1037


# todo train model dziala tymczasowo na tablicy stringów zaiast tablicy 3 wym int !!!!
# todo dodaj 3ci alg
# todo reszta wariantów 2 filmików
# todo przepisanie wyników


# train model jest na ciagu pkt ze wszystkich algorytmow, ze wszystkich filmow?
# czyli X_train = BPf10+OPf10+DPf10+BPf20+OPf20+DPf20 i Y_train = expf1+expf1+expf1+expf2+expf2+expf2
def train_model(data, expected):
    # x data, y labels
    # train = expected, X = data. Y = labels
    X_train = data
    Y_train = expected
    # train model
    vectorizer = TfidfVectorizer()
    regressor = LogisticRegression()
    pipeline = Pipeline([('vectorizer', vectorizer), ('regressor', regressor)])
    pipeline.fit(X_train, Y_train)
    return pipeline


def predict_pose(data):
    global expected
    # x data, y labels
    # train = expected, X = data. Y = labels
    X_train = data
    Y_train = expected
    X_validation = data
    # train model
    vectorizer = TfidfVectorizer()
    regressor = LogisticRegression()
    pipeline = Pipeline([('vectorizer', vectorizer), ('regressor', regressor)])
    pipeline.fit(X_train, Y_train)
    # predict
    result = pipeline.predict(X_validation)  # value in '0','1', etc
    return result


def get_BP_data(loc):
    global poses_total
    # read from xls
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    point_nr = 0
    all_frames = []
    for i in range(poses_total):  # row
        i += 1
        row = ''
        for j in range(32):  # column
            cell = sheet.cell_value(i, j)
            if len(cell) > 0:
                point_nr += 1
            row += cell
        all_frames.append(row)
    return all_frames, point_nr


def get_OP_data(loc):
    global poses_total
    # read from json
    point_nr = 0
    all_frames = []
    for i in range(poses_total):
        nr = str(i).zfill(12)
        # "000000000000"
        filename = loc + nr + "_keypoints.json"
        with open(filename) as json_file:
            row = []
            data = json.load(json_file)
            data = data['people']
            if data != []:
                data = data[0]['pose_keypoints_2d']
            for j in range(0, len(data), 3):  # 75 łącznie c[74],y[73],x[72]
                # [x0,y0,c0,x1,y1,c1...]
                cell = [data[j], data[j + 1]]
                point_nr += 1
                row.append(cell)
        all_frames.append(str(row))
    return all_frames, point_nr


def result_statistics(poses):
    global expected, poses_total
    point_nr = 0
    correct_poses = 0
    frame = 0
    startframe = frame

    nr_stand = 0
    nr_sit = 0
    nr_lay = 0
    nr_bow = 0
    nr_und = 0
    cr_stand = 0
    cr_sit = 0
    cr_lay = 0
    cr_bow = 0
    cr_und = 0

    # statistics of correctness
    for i in range(poses_total):  # row
        if len(poses) > 1 and poses[i] != poses[i - 1]:
            correct = 0
            for k in range(frame - startframe):
                if str(poses[k + startframe]) == expected[k + startframe]:
                    correct += 1

            if poses[i - 1] == '1':
                # print("standing: (", startframe, frame, ") correct:", correct, '/', frame - startframe)
                cr_stand += correct
                nr_stand += frame - startframe
            elif poses[i - 1] == '2':
                # print("sitting: (", startframe, frame, ") correct:", correct, '/', frame - startframe)
                cr_sit += correct
                nr_sit += frame - startframe
            elif poses[i - 1] == '3':
                # print("laying: (", startframe, frame, ") correct:", correct, '/', frame - startframe)
                cr_lay += correct
                nr_lay += frame - startframe
            elif poses[i - 1] == '4':
                # print("bowing: (", startframe, frame, ") correct:", correct, '/', frame - startframe)
                cr_bow += correct
                nr_bow += frame - startframe
            else:
                # print("undefined: (", startframe, frame, ") correct:", correct, '/', frame - startframe)
                cr_und += correct
                nr_und += frame - startframe
            startframe = frame
        frame += 1

        # verify_poses
        if poses[i] == expected[i]:
            correct_poses += 1

    #print("Total correctness:")
    #print("Standing: ", cr_stand / nr_stand)
    #print("Sitting: ", cr_sit / nr_sit)
    #print("Laying: ", cr_lay / nr_lay)
    #print("Bowing: ", cr_bow / nr_bow)
    #print("Undefined: ", cr_und, nr_und)

    return correct_poses


def write_results(row, point_nr, percent):
    df1 = pd.read_excel("Results.xlsx")
    df1.at[row, 'ilość punktów'] = point_nr
    df1.at[row, 'Poprawność klasyfikatora'] = percent
    df1.to_excel("Results.xlsx", index=False)

def predict_BP(model,filename):
    BP_data, BP_pnr = get_BP_data(filename+".xls")
    BP_result = model.predict(BP_data)
    correct_poses = result_statistics(BP_result)
    print("\npoints in",filename,":", BP_pnr)
    print("correct poses in",filename,":", correct_poses, '/', poses_total, ",", int(correct_poses / poses_total * 100),
          '%')
    write_results(0, BP_pnr, correct_poses / poses_total)

def predict_OP(model,filename):
    OP_data, OP_pnr = get_OP_data(filename)
    OP_result = model.predict(OP_data)
    correct_poses = result_statistics(OP_result)
    print("\npoints in",filename,":", OP_pnr)
    print("correct poses in",filename,":", correct_poses, '/', poses_total, ",", int(correct_poses / poses_total * 100),
          '%')
    write_results(6, OP_pnr, correct_poses / poses_total)

# main function
def main():
    # expected f1
    global expected, poses_total
    file1 = open("mat_color_expected.txt", "r+")
    list = file1.read()
    for result in list:
        expected.append(result)

    # get data f1
    BP_data, BP_pnr = get_BP_data("BP/BP_f10.xls")
    OP_data, OP_pnr = get_OP_data('OP/OP_f10/mat_color_')
    model = train_model(BP_data + OP_data, expected + expected)

    print("\nBlazePose")
    predict_BP(model,"BP/BP_f10")

    print("\nOpenPose")
    predict_OP(model,'OP/OP_f10/mat_color_')
    # DeepPose


if __name__ == '__main__':
    data = [
        [[1, 2], [1, 2], [1, 2]],
        [[1, 1], [1, 1], [1, 1]],
        [[2, 2], [2, 2], [2, 2]]
    ]
    expect=[1,2,3]
    #train_model(data, expect)
    main()
