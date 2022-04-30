import xlsxwriter
import xlrd

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

expected = []


# stand 375
# sit 281
# lay 202
# bow 179
# undefined 0

def predict_pose(data):
    global expected
    # x data, y labels
    # train = expected. X data. Y labels
    X_train = data
    Y_train = expected[:1028]
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
    # read from xls
    global expected
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    all_frames = []
    for i in range(1028):  # row
        i += 1
        row = []
        for j in range(32):  # column
            cell = sheet.cell_value(i, j)
            cell = cell[1:len(cell) - 1]
            cell = cell.split(',')
            cell = [int(cell[0]), int(cell[1])]
            row.append(cell)
        all_frames.append(str(row))
    return all_frames


def read_excel():
    loc = ("Example2.xls")
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    print(sheet.cell_value(0, 0))
    for i in range(11):  # row
        expected.append(1)


def write_to_excel():
    book = xlsxwriter.Workbook('Example2.xlsx')
    sheet = book.add_worksheet()
    # Rows and columns are zero indexed.
    row = 0
    column = 0
    content = [["Parker", "Smith", "John"], ["john", "a", "b"], [1, 2, 3]]
    # iterating through the content list
    for item in content:
        for cell in item:
            sheet.write(row, column, cell)
            column += 1
        column = 0
        row += 1
    book.close()


def measure_distance_XY(point1, point2):
    disX = point2[0] - point1[0]
    disY = point2[1] - point1[1]
    if disX > disY:
        return 0
    else:
        return 1


# 0=?, 1=stand, 2=sit, 3=lay, 4=bow
# 57%
def interpret_pose(nose, hips, knees, feet):
    # measure distance between joints
    # 0 = x distance longer, 1 y distance longer
    n_to_h = measure_distance_XY(nose, hips)
    h_to_k = measure_distance_XY(hips, knees)
    k_to_f = measure_distance_XY(knees, feet)

    # standing = nose>hips>knees>feet
    if n_to_h == 1 and h_to_k == 1 and k_to_f == 1 and (nose[1] < hips[1] < knees[1] < feet[1]) and (
            feet[1] - nose[1] > 700) and int(hips[0] / 100) == int(nose[0] / 100):
        return 1
    # bowing = nose<hips>knees>feet
    elif k_to_f == 1 and (n_to_h == 0 or int(hips[1] / 100) == int(nose[1] / 100)):
        return 4
    # sitting = nose>hips=knees>feet
    elif n_to_h == 1 and int(hips[1] / 100) == int(feet[1] / 100):
        return 2
    # laying = nose=hips=knees
    elif (n_to_h == 0 and h_to_k == 0 and k_to_f == 0) or (n_to_h == 1 and h_to_k == 1 and k_to_f == 1):
        return 3
    else:
        return 0


#  10%
def interpret_pose_old(nose, hips, knees, feet):
    nose = [int(nose[0] / 10), int(nose[1] / 10)]
    hips = [int(hips[0] / 10), int(hips[1] / 10)]
    knees = [int(knees[0] / 10), int(knees[1] / 10)]
    feet = [int(feet[0] / 10), int(feet[1] / 10)]

    # standing = nose>hips>knees>feet
    if (nose[1] < hips[1] < knees[1] < feet[1]) and (int(nose[0] / 10) == int(hips[0] / 10) == int(knees[0] / 10)):
        return 1
    # sitting = nose>hips=knees>feet
    elif (nose[1] < hips[1] and int(hips[1] / 10) == int(knees[1] / 10) and knees[1] < feet[1]):
        return 2
    # laying = nose=hips=knees
    elif (int(nose[1] / 10) == int(hips[1] / 10) == int(knees[1] / 10)) and (nose[0] < hips[0] < knees[0] < feet[0]):
        return 3
    # bowing = nose<hips>knees>feet
    elif (nose[1] > hips[1] < knees[1] < feet[1]):
        return 4
    else:
        return 0


def result_statistics(poses):
    global expected
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
    for i in range(1028):  # row
        if len(poses) > 1 and poses[i] != poses[i - 1]:
            correct = 0
            for k in range(frame - startframe):
                if str(poses[k + startframe]) == expected[k + startframe]:
                    correct += 1

            if poses[i - 1] == '1':
                print("standing: (", startframe, frame, ") correct:", correct, '/', frame - startframe)
                cr_stand += correct
                nr_stand += frame - startframe
            elif poses[i - 1] == '2':
                print("sitting: (", startframe, frame, ") correct:", correct, '/', frame - startframe)
                cr_sit += correct
                nr_sit += frame - startframe
            elif poses[i - 1] == '3':
                print("laying: (", startframe, frame, ") correct:", correct, '/', frame - startframe)
                cr_lay += correct
                nr_lay += frame - startframe
            elif poses[i - 1] == '4':
                print("bowing: (", startframe, frame, ") correct:", correct, '/', frame - startframe)
                cr_bow += correct
                nr_bow += frame - startframe
            else:
                print("undefined: (", startframe, frame, ") correct:", correct, '/', frame - startframe)
                cr_und += correct
                nr_und += frame - startframe
            startframe = frame
        frame += 1

        # verify_poses
        if (poses[i] == expected[i]):
            correct_poses += 1

    print("\nTotal correctness:")
    print("Standing: ", cr_stand, nr_stand)
    print("Sitting: ", cr_sit, nr_sit)
    print("Laying: ", cr_lay, nr_lay)
    print("Bowing: ", cr_bow, nr_bow)
    print("Undefined: ", cr_und, nr_und)

    return point_nr, correct_poses, i


def check_pose(loc):
    global expected

    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)

    point_nr = 0
    correct_poses = 0

    frame = 0
    startframe = frame
    poses = []

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

    for i in range(1028):  # row
        i += 1
        row = []
        for j in range(32):  # column
            cell = sheet.cell_value(i, j)
            cell = cell[1:len(cell) - 1]
            cell = cell.split(',')
            cell = [int(cell[0]), int(cell[1])]
            row.append(cell)
        point_nr += len(row)
        pose = interpret_pose(row[0], row[23], row[25], row[27])
        poses.append(pose)
        if len(poses) > 1 and pose != poses[-2]:
            correct = 0
            for k in range(frame - startframe):
                if str(poses[k + startframe]) == expected[k + startframe]:
                    correct += 1

            if poses[-2] == 1:
                print("standing: (", startframe, frame, ") correct:", correct, '/', frame - startframe)
                cr_stand += correct
                nr_stand += frame - startframe
            elif poses[-2] == 2:
                print("sitting: (", startframe, frame, ") correct:", correct, '/', frame - startframe)
                cr_sit += correct
                nr_sit += frame - startframe
            elif poses[-2] == 3:
                print("laying: (", startframe, frame, ") correct:", correct, '/', frame - startframe)
                cr_lay += correct
                nr_lay += frame - startframe
            elif poses[-2] == 4:
                print("bowing: (", startframe, frame, ") correct:", correct, '/', frame - startframe)
                cr_bow += correct
                nr_bow += frame - startframe
            else:
                print("undefined: (", startframe, frame, ") correct:", correct, '/', frame - startframe)
                cr_und += correct
                nr_und += frame - startframe
            startframe = frame
        frame += 1

        # verify_poses
        if (str(pose) == expected[i - 1]):
            correct_poses += 1

    print("\nTotal correctness:")
    print("Standing: ", cr_stand, nr_stand)
    print("Sitting: ", cr_sit, nr_sit)
    print("Laying: ", cr_lay, nr_lay)
    print("Bowing: ", cr_bow, nr_bow)
    print("Undefined: ", cr_und, nr_und)

    return point_nr, correct_poses, i


def import_files():
    global expected
    file1 = open("mat_color_expected.txt", "r+")
    list = file1.read()
    for result in list:
        expected.append(result)

    # BlazePose
    # point_nr, correct_poses, poses_total = check_pose("BP_f10.xls")
    BP_data = get_BP_data("BP_f10.xls")
    result = predict_pose(BP_data)
    point_nr, correct_poses, poses_total = result_statistics(result)
    print("\npoints in f10 BP: ", point_nr)
    print("correct poses in f10 BP: ", correct_poses, '/', poses_total, ",", int(correct_poses / poses_total * 100),
          '%')
    # write to excel

    # OpenPose

    # DeepPose


if __name__ == '__main__':
    import_files()
