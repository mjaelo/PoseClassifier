import json
import xlrd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

expected = []


# todo train model zrobic na ciagu pkt ze wszystkich algorytmow, ze wszystkich filmow?
# czyli X_train = BPf10+OPf10+DPf10+BPf20+OPf20+DPf20 i Y_train = expf1+expf1+expf1+expf2+expf2+expf2

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
    # read from xls
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    all_frames = []
    for i in range(1037):  # row
        i += 1
        row = ''
        for j in range(32):  # column
            cell = sheet.cell_value(i, j)
            row += cell
        all_frames.append(row)
    return all_frames


def get_OP_data(loc):
    # read from json
    all_frames = []
    for i in range(1037):
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
                row.append(cell)
        all_frames.append(str(row))
    return all_frames


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


def import_files():
    global expected
    file1 = open("mat_color_expected.txt", "r+")
    list = file1.read()
    for result in list:
        expected.append(result)

    # BlazePose
    BP_data = get_BP_data("BP_f10.xls")
    BP_result = predict_pose(BP_data)
    point_nr, correct_poses, poses_total = result_statistics(BP_result)
    print("\npoints in f10 BP: ", point_nr)
    print("correct poses in f10 BP: ", correct_poses, '/', poses_total, ",", int(correct_poses / poses_total * 100),
          '%')
    # write to excel

    # OpenPose
    OP_data = get_OP_data('OP_f10/mat_color_')
    OP_result = predict_pose(OP_data)
    point_nr, correct_poses, poses_total = result_statistics(OP_result)
    print("\npoints in f10 OP: ", point_nr)
    print("correct poses in f10 OP: ", correct_poses, '/', poses_total, ",", int(correct_poses / poses_total * 100),
          '%')
    # write to excel

    # DeepPose


if __name__ == '__main__':
    import_files()
