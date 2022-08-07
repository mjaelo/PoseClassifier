from collections import Counter

import cv2
import numpy as np

movie = "f2_normal"
cap = cv2.VideoCapture("videos/"+movie + '.mp4')


# mat_color  nr of frames:  1037
# mat_white  nr of frames:  1073


def read_txt():
    file1 = open("expected/" + movie + "_expected.txt", "r+")
    results = file1.read()
    results=  [char for char in results]
    nr = results.count('1')
    print("stand", nr)
    nr = results.count('2')
    print("sit", nr)
    nr = results.count('3')
    print("lay", nr)
    nr = results.count('4')
    print("bow", nr)
    nr = results.count('0')
    print("undefined", nr)


def count_avg(lines):
    avg_line = ''
    for i in range(len(lines[0]) - 1):
        frame_poses = []
        for li in lines:
            frame_poses.append(li[i])
        occurence_count = Counter(frame_poses)
        most_common = occurence_count.most_common(1)[0][0]
        avg_line += most_common
    print("avg:", avg_line)
    return avg_line


def append_result(new_line):
    file1 = open("expected/" + movie + "_all.txt", "r+")
    line = "temp"
    all_lines = []
    while line != "":
        line = file1.readline()
        if line != "":
            all_lines.append(line)
            print(line)
    all_lines.append(new_line+ "\n")
    file1 = open("expected/" + movie + "_all.txt", "w")
    file1.writelines(all_lines)
    file1.close()
    #print(all_lines)
    return all_lines


def play_check():
    nr = 0
    key = ord('a')
    ret, img = cap.read()

    # resize image
    width = int(img.shape[1] / 2)
    height = int(img.shape[0] / 2)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    cv2.putText(img, str(nr), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 8, 8), 3)
    nr += 1
    go = 0
    results_list = ''
    try:
        # 0=?, 1=stand, 2=sit, 3=lay, 4=bow
        while key != ord('q'):
            if key == ord(' '):
                go = 1
                results_list += '0'
            elif key == ord('1'):
                go = 1
                results_list += '1'
            elif key == ord('2'):
                go = 1
                results_list += '2'
            elif key == ord('3'):
                go = 1
                results_list += '3'
            elif key == ord('4'):
                go = 1
                results_list += '4'
            else:
                go = 0
            if go == 1:
                ret, img = cap.read()
                # resize image
                width = int(img.shape[1] / 2)
                height = int(img.shape[0] / 2)
                dim = (width, height)
                img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

                cv2.putText(img, str(nr), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 8, 8), 3)
                nr += 1
            cv2.imshow('frame', img)
            key = cv2.waitKey(30)
    except:
        print("nr of frames: ", nr)

        all_lines = append_result(results_list)#all lines
        results_list = count_avg(all_lines)

        file1 = open("expected/" + movie + "_expected.txt", "w")
        file1.writelines(results_list)
        file1.close()
        print("saved frames:", len(results_list))
        cap.release()
        cv2.destroyAllWindows()
        read_txt()


if __name__ == '__main__':
    play_check()
    # read_txt()
