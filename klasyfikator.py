import xlsxwriter
import xlrd

expected = []


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


# 0=?, 1=stand, 2=sit, 3=lay, 4=bow
def interpret_pose(nose, hips,  knees, feet):
    # standing = nose>hips>knees>feet
    if nose[1] < hips[1] < knees[1] < feet[1]:
        return 1
    # sitting = nose>hips=knees>feet
    elif nose[1] < hips[1] == knees[1] < feet[1]:
        return 2
    # laying = nose=hips=knees
    elif nose[1] == hips[1] == knees[1]:
        return 3
    # bowing = nose<hips>knees>feet
    elif nose[1] > hips[1] < knees[1] < feet[1]:
        return 4
    else:
        return 0


def get_pose(loc):
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

            if pose == 1:
                print("standing: (", startframe, frame, ") correct:", correct, '/', frame - startframe)
                cr_stand += correct
                nr_stand += frame - startframe
            elif pose == 2:
                print("sitting: (", startframe, frame, ") correct:", correct, '/', frame - startframe)
                cr_sit += correct
                nr_sit += frame - startframe
            elif pose == 3:
                print("laying: (", startframe, frame, ") correct:", correct, '/', frame - startframe)
                cr_lay += correct
                nr_lay += frame - startframe
            elif pose == 4:
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
    point_nr, correct_poses, poses_total = get_pose("BP_f10.xls")
    print("\npoints in f10 BP: ", point_nr)
    print("correct poses in f10 BP: ", correct_poses, '/', poses_total)
    # write to excel

    # OpenPose

    # DeepPose


if __name__ == '__main__':
    import_files()
