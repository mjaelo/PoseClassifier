from collections import Counter

import cv2
import numpy as np

movie = "mat_color"
cap = cv2.VideoCapture(movie + '.mp4')


# mat_color  nr of frames:  1037


# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.mp4', -1, 20.0, (640, 480))


def append_txt(l1):
    file1 = open(movie + "_all.txt", "r+")
    l1 = file1.read() + l1

    file1 = open(movie + "_all.txt", "w")
    file1.writelines(l1 + "\n")
    file1.close()
    cap.release()
    cv2.destroyAllWindows()

    file1 = open(movie + "_all.txt", "r+")
    line = "temp"
    lines = []
    while line != "":
        line = file1.readline()
        if line != "":
            lines.append(line)
            print(line)
    return count_avg(lines)


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


def read_txt():
    file1 = open(movie + "_expected.txt", "r+")
    list = file1.read()
    nr = list.count('1')
    print("stand", nr)
    nr = list.count('2')
    print("sit", nr)
    nr = list.count('3')
    print("lay", nr)
    nr = list.count('4')
    print("bow", nr)
    nr = list.count('0')
    print("undefined", nr)


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
    list = ''
    try:
        # 0=?, 1=stand, 2=sit, 3=lay, 4=bow
        while key != ord('q'):
            if key == ord(' '):
                go = 1
                list += '0'
            elif key == ord('1'):
                go = 1
                list += '1'
            elif key == ord('2'):
                go = 1
                list += '2'
            elif key == ord('3'):
                go = 1
                list += '3'
            elif key == ord('4'):
                go = 1
                list += '4'
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

        list = append_txt(list)

        file1 = open(movie + "_expected.txt", "w")
        file1.writelines(list)
        file1.close()
        print("saved frames:", len(list))
        cap.release()
        cv2.destroyAllWindows()
        read_txt()

if __name__ == '__main__':
    play_check()
    # read_txt()
