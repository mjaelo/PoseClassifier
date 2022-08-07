import json
import os
from cmath import nan

import xlrd
import pandas as pd

import sklearn.feature_extraction.image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

expected1 = []
poses_total_f1 = 1037
expected2 = []
poses_total_f2 = 1073


# todo train model dziala tymczasowo na tablicy stringów zaiast tablicy 3 wym int
# todo dziwne że AP nie ma brakujacych klatek
# todo może dac do porównania 3 modele dla kazdego alg?

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


# po co mi points_nr, jak jest problem chyba tylko z pustymi klatkami?
# gdy jest puste, to czyta jako [nan,nan,nan]
# dlugosc 33 punkty an klatke
# zwraca dane w formacie [ '[[x,y],[],...]', '[[],[]]', ... ]
def get_BP_data(loc):
    global poses_total_f1, poses_total_f2
    poses_total = poses_total_f1
    if loc[-2] == '2':
        poses_total = poses_total_f2

    # read from xlsx
    df1 = pd.read_excel(loc + ".xlsx")
    all = df1.values.tolist()
    while len(all)!=poses_total:
        all.append([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan])
    all_frames = []
    empty_frames = 0#poses_total - len(all)-1

    for fr in all:
        frame = [fr[0], fr[11],  # glowa
                 fr[12], fr[14], fr[16],  # lreka
                 fr[11], fr[13], fr[15],  # preka
                 fr[24], fr[26], fr[28],  # lnoga
                 fr[23], fr[25], fr[27],  # pnoga
                 fr[5], fr[2], fr[8], fr[7]  # oczy
                 ]
        # pkt na pozycji 1 jest zły. jest naprawde pomiedzy 11 i 12
        is_nan = 0
        row = ''
        for i in frame:
            i = str(i)
            if len(all_frames)==poses_total:
                break
            elif i == 'nan':
                is_nan = 1
                break  # nwm jak program reaguje na klatke, gdzie jest tylko jedno 'nan', wiec wole jescze nie zmieniac niczego
            else:
                i = i.replace(' ', '')
            tempi = ''
            add = 1
            # zaokreglenie do 3 miejsc poo przecinku
            for j in range(len(i)):
                if add == 1:
                    tempi += i[j]
                if i[j] == '.':
                    add = 0
                    tempi += i[j + 1]
                    if len(i)>j+2:
                        tempi += i[j + 2]
                        if len(i) > j + 3:
                            tempi += i[j + 3]
                if i[j] == ',':
                    add = 1
                    tempi += i[j]
                    j += 1
            row += tempi + ']' + ','
        if is_nan == 1:
            empty_frames += 1
        if len(all_frames) != poses_total:
            all_frames.append('[' + row + ']')

    empty = df1.isna().sum().sum()  # brakujące punkty
    point_nr = poses_total * 32 - empty
    return all_frames, empty_frames


# gdy puste, cały plik to {"version":1.3,"people":[]}
# f10 ma 1036 pliki a f1d tylko 1034, co powoduje błedy. dodac je do brakujących klatek?
# dlugosc 25 punkty an klatke
def get_OP_data(loc):
    global poses_total_f1, poses_total_f2
    poses_total = poses_total_f1
    if loc[7] == '2':  # wybór filmiku
        poses_total = poses_total_f2

    # read from json
    point_nr = 0
    all_frames = []
    empty_frames = 0
    for i in range(poses_total):
        nr = str(i).zfill(12)
        # "000000000000"
        filename = loc + nr + "_keypoints.json"
        if os.path.exists(filename) == False:
            # print("missing file: " + filename)
            empty_frames += 1
        else:
            with open(filename) as json_file:
                row = []
                data = json.load(json_file)
                data = data['people']
                if data == []:
                    empty_frames += 1
                if data != []:
                    data = data[0]['pose_keypoints_2d']
                for j in range(0, len(data), 3):  # 75 łącznie c[74],y[73],x[72]
                    # [x0,y0,c0,x1,y1,c1...]
                    x = round(data[j], 3)
                    y = round(data[j + 1], 3)
                    cell = [x, y]
                    point_nr += 1
                    row.append(cell)
        fr = row
        row_AP=[]
        if len(fr) > 17:
            row_AP = [fr[0], fr[1],  # glowa
                      fr[2], fr[3], fr[4],  # lreka
                      fr[5], fr[6], fr[7],  # preka
                      fr[9], fr[10], fr[11],  # lnoga
                      fr[12], fr[13], fr[14],  # pnoga
                      fr[5], fr[16], fr[17], fr[18]  # oczy
                      ]
        all_frames.append(str(row_AP))
    return all_frames, empty_frames


# 17 keypointsow w jednej klatce
def get_AP_data(loc):
    global poses_total_f1, poses_total_f2
    poses_total = poses_total_f1
    if loc[7] == '2':
        poses_total = poses_total_f2

    # read from json
    point_nr = 0
    all_frames = []
    empty_frames = 0
    filename = loc + ".json"
    with open(filename) as json_file:
        data = json.load(json_file)
        for i in range(poses_total):
            frame = data[i]['keypoints']
            row = []
            if len(frame) == 0:
                empty_frames += 1
            for j in range(0, len(frame), 3):  # 75 łącznie c[74],y[73],x[72]
                # [x0,y0,c0,x1,y1,c1...]

                x = round(frame[j], 3)
                y = round(frame[j + 1], 3)
                cell = [x, y]
                point_nr += 1
                row.append(cell)
            all_frames.append(str(row))
    return all_frames, empty_frames


def result_statistics(poses,film_nr):
    global expected1,expected2, poses_total_f1,poses_total_f2

    poses_total = poses_total_f1
    expected = expected1
    if film_nr == '2':
        poses_total = poses_total_f2
        expected=expected2

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

    #print("\nTotal correctness:")
    #print("Standing: ", cr_stand / nr_stand)
    #print("Sitting: ", cr_sit / nr_sit)
    #print("Laying: ", cr_lay / nr_lay)
    #print("Bowing: ", cr_bow / nr_bow)
    # print("Undefined: ", cr_und, nr_und)

    return correct_poses


def write_results(row, missing_frames, percent):
    df1 = pd.read_excel("Results.xlsx")
    df1.at[row, 'Brakujące klatki'] = missing_frames
    df1.at[row, 'Poprawność klasyfikatora'] = percent
    df1.to_excel("Results.xlsx", index=False)


def predict_pose_old(model, filename):
    if filename[0] == 'B':
        data, missing_frames = get_BP_data(filename)
    if filename[0] == 'O':
        data, missing_frames = get_OP_data(filename)
    if filename[0] == 'A':
        data, missing_frames = get_AP_data(filename)
    result = model.predict(data)
    correct_poses = result_statistics(result)

    print("missing frames in", filename, ":", missing_frames)

    global poses_total_f1, poses_total_f2
    if filename[7] == '1':
        print("correct poses in", filename, ":", correct_poses, '/', poses_total_f1, ",",
              int(correct_poses / poses_total_f1 * 100),
              '%')
        return missing_frames, correct_poses / poses_total_f1
    else:
        print("correct poses in", filename, ":", correct_poses, '/', poses_total_f2, ",",
              int(correct_poses / poses_total_f2 * 100),
              '%')
        return missing_frames, correct_poses / poses_total_f2


def predict_pose(model, filename, nr):
    if filename[0] == 'B':
        data, missing_frames = get_BP_data(filename)
    if filename[0] == 'O':
        data, missing_frames = get_OP_data(filename)
    if filename[0] == 'A':
        data, missing_frames = get_AP_data(filename)
    result = model.predict(data)
    correct_poses = result_statistics(result,filename[7])

    print("missing frames in", filename, ":", missing_frames)

    global poses_total_f1, poses_total_f2
    if filename[7] == '1':
        print("correct poses in", filename, ":", correct_poses, '/', poses_total_f1, ",",
              int(correct_poses / poses_total_f1 * 100),
              '%')
        correctness = correct_poses / poses_total_f1
    else:
        print("correct poses in", filename, ":", correct_poses, '/', poses_total_f2, ",",
              int(correct_poses / poses_total_f2 * 100),
              '%')
        correctness = correct_poses / poses_total_f2

    # todo write_results(nr, missing_frames, correctness)


# main function
def main():
    do_what = [1, 1, 1, 1, 1, 1]  # BPf1, BPf2, OPf1, OPf2, APf1, APf2
    # expected f1
    global expected1
    file1 = open("expected/f1_normal_expected.txt", "r+")
    list1 = file1.read()
    for result in list1:
        expected1.append(result)
    # expected f2
    global expected2
    file2 = open("expected/f2_normal_expected.txt", "r+")
    list2 = file2.read()
    for result in list2:
        expected2.append(result)

    # get X_train data
    BP_data1, _ = get_BP_data("BP/BP_f10")
    BP_data2, _ = get_BP_data("BP/BP_f20")
    OP_data1, _ = get_OP_data('OP/OP_f10/f1_normal_')
    OP_data2, _ = get_OP_data('OP/OP_f20/f2_normal_')
    AP_data1, _ = get_AP_data("AP/AP_f10")
    AP_data2, _ = get_AP_data("AP/AP_f20")

    # get Y_train data
    X_train = BP_data1 + BP_data2 + OP_data1 + OP_data2 + AP_data1 + AP_data2
    Y_train = expected1 + expected2 + expected1 + expected2 + expected1 + expected2

    for i in range(len(X_train)):
        if i < len(X_train):
            if X_train[i] == '[]':
                del X_train[i]
                del Y_train[i]
    model = train_model(X_train, Y_train)

    # use model for getting data
    print("\nBlazePose")
    if do_what[0] == 1:
        numbers = [0, 1, 2, 3]
        filenames = ["BP/BP_f10", "BP/BP_f1d", "BP/BP_f1b", "BP/BP_f1a"]
        for i in range(len(numbers)):
            predict_pose(model, filenames[i], numbers[i])
    if do_what[1] == 1:
        numbers = [4, 5, 6, 7]
        filenames = ["BP/BP_f20", "BP/BP_f2d", "BP/BP_f2b", "BP/BP_f2a"]
        for i in range(len(numbers)):
            predict_pose(model, filenames[i], numbers[i])

    print("\nOpenPose")
    if do_what[2] == 1:
        numbers = [8, 9, 10, 11]
        filenames = ['OP/OP_f10/f1_normal_', 'OP/OP_f1d/f1_dark_', 'OP/OP_f1b/f1_black_', 'OP/OP_f1a/f1_all_']
        for i in range(len(numbers)):
            predict_pose(model, filenames[i], numbers[i])
    if do_what[3] == 1:
        numbers = [12, 13, 14, 15]
        filenames = ['OP/OP_f20/f2_normal_', 'OP/OP_f2d/f2_dark_', 'OP/OP_f2b/f2_black_', 'OP/OP_f2a/f2_all_']
        for i in range(len(numbers)):
            predict_pose(model, filenames[i], numbers[i])

    print("\nAlphaPose")
    if do_what[4] == 1:
        numbers = [16, 17, 18, 19]
        filenames = ["AP/AP_f10", "AP/AP_f1d", "AP/AP_f1b", "AP/AP_f1a"]
        for i in range(len(numbers)):
            predict_pose(model, filenames[i], numbers[i])
    if do_what[5] == 1:
        numbers = [20, 21, 22, 23]
        filenames = ["AP/AP_f20", "AP/AP_f2d", "AP/AP_f2b", "AP/AP_f2a"]
        for i in range(len(numbers)):
            predict_pose(model, filenames[i], numbers[i])


def odl():
    model = []
    do_what = [0, 0, 0, 0, 0, 0, 0]
    # use model for getting data
    print("\nBlazePose")
    if do_what[0] == 1:
        BP_missing_frames, correctness = predict_pose_old(model, "BP/BP_f10")
        write_results(0, BP_missing_frames, correctness)
        BP_missing_frames, correctness = predict_pose_old(model, "BP/BP_f1d")
        write_results(1, BP_missing_frames, correctness)
        BP_missing_frames, correctness = predict_pose_old(model, "BP/BP_f1b")
        write_results(2, BP_missing_frames, correctness)
        BP_missing_frames, correctness = predict_pose_old(model, "BP/BP_f1a")
        write_results(3, BP_missing_frames, correctness)
    if do_what[1] == 1:
        BP_missing_frames, correctness = predict_pose_old(model, "BP/BP_f20")
        write_results(4, BP_missing_frames, correctness)
        BP_missing_frames, correctness = predict_pose_old(model, "BP/BP_f2d")
        write_results(5, BP_missing_frames, correctness)
        BP_missing_frames, correctness = predict_pose_old(model, "BP/BP_f2b")
        write_results(6, BP_missing_frames, correctness)
        BP_missing_frames, correctness = predict_pose_old(model, "BP/BP_f2a")
        write_results(7, BP_missing_frames, correctness)

    print("\nOpenPose")
    if do_what[2] == 1:
        OP_missing_frames, correctness = predict_pose_old(model, 'OP/OP_f10/f1_normal_')
        write_results(8, OP_missing_frames, correctness)
        OP_missing_frames, correctness = predict_pose_old(model, 'OP/OP_f1d/f1_dark_')
        write_results(9, OP_missing_frames, correctness)
        OP_missing_frames, correctness = predict_pose_old(model, 'OP/OP_f1b/f1_black_')
        write_results(10, OP_missing_frames, correctness)
        OP_missing_frames, correctness = predict_pose_old(model, 'OP/OP_f1a/f1_all_')
        write_results(11, OP_missing_frames, correctness)
    if do_what[3] == 1:
        OP_missing_frames, correctness = predict_pose_old(model, 'OP/OP_f20/f2_normal_')
        write_results(12, OP_missing_frames, correctness)
        OP_missing_frames, correctness = predict_pose_old(model, 'OP/OP_f2d/f2_dark_')
        write_results(13, OP_missing_frames, correctness)
        OP_missing_frames, correctness = predict_pose_old(model, 'OP/OP_f2b/f2_black_')
        write_results(14, OP_missing_frames, correctness)
        OP_missing_frames, correctness = predict_pose_old(model, 'OP/OP_f2a/f2_all_')
        write_results(15, OP_missing_frames, correctness)

    print("\nAlphaPose")
    if do_what[4] == 1:
        AP_missing_frames, correctness = predict_pose_old(model, "AP/AP_f10")
        write_results(16, AP_missing_frames, correctness)
        AP_missing_frames, correctness = predict_pose_old(model, "AP/AP_f1d")
        write_results(17, AP_missing_frames, correctness)
        AP_missing_frames, correctness = predict_pose_old(model, "AP/AP_f1b")
        write_results(18, AP_missing_frames, correctness)
        AP_missing_frames, correctness = predict_pose_old(model, "AP/AP_f1a")
        write_results(19, AP_missing_frames, correctness)
    if do_what[5] == 1:
        AP_missing_frames, correctness = predict_pose_old(model, "AP/AP_f20")
        write_results(20, AP_missing_frames, correctness)
        AP_missing_frames, correctness = predict_pose_old(model, "AP/AP_f2d")
        write_results(21, AP_missing_frames, correctness)
        AP_missing_frames, correctness = predict_pose_old(model, "AP/AP_f2b")
        write_results(22, AP_missing_frames, correctness)
        AP_missing_frames, correctness = predict_pose_old(model, "AP/AP_f2a")
        write_results(23, AP_missing_frames, correctness)


if __name__ == '__main__':
    data = [
        [[1, 2], [1, 2], [1, 2]],
        [[1, 1], [1, 1], [1, 1]],
        [[2, 2], [2, 2], [2, 2]]
    ]
    expect = [1, 2, 3]

    # train_model(data, expect)
    main()
