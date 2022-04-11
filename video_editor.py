import cv2
import numpy as np

movie="mat_color"
cap = cv2.VideoCapture(movie+'.mp4')
# mat_color  nr of frames:  1037


# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.mp4', -1, 20.0, (640, 480))

def read_txt():
    file1 = open("maunal_results.txt", "r+")
    print(file1.read())

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
    list=[]
    try:
        # 0=?, 1=stand, 2=sit, 3=lay, 4=bow
        while key != ord('q'):
            if key == ord(' '):
                go=1
                list.append('0')
            elif key == ord('1'):
                go=1
                list.append('1')
            elif key == ord('2'):
                go=1
                list.append('2')
            elif key == ord('3'):
                go=1
                list.append('3')
            elif key == ord('4'):
                go=1
                list.append('4')
            else:
                go=0
            if go==1:
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
        file1 = open(movie+"_expected.txt", "w")
        file1.writelines(list)
        file1.close()

        print("saved frames:",len(list))
        cap.release()
        cv2.destroyAllWindows()


def play_frame():
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

    go=0
    while key != ord('q'):
        if key == ord(' '):
            ret, img = cap.read()

            # resize image
            width = int(img.shape[1]/2)
            height = int(img.shape[0]/2)
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

            cv2.putText(img, str(nr), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 8, 8), 3)
            nr += 1
        cv2.imshow('frame', img)
        key = cv2.waitKey(30)
    print("nr of frames: ", nr)
    cap.release()
    cv2.destroyAllWindows()


def play():
    nr = 0

    while cap.isOpened():
        success, img = cap.read()
        if success:
            # cv2.rectangle(img, (384, 500), (1000, 128), (0, 0, 0), -1)
            # for i in range(len(img)):
            # for j in range(len(img[0])):
            # p1 = img[i][j]
            # img[i][j] = [p1[0] / 2, p1[1] / 2, p1[2] / 2]
            # out.write(img)
            nr += 1
            cv2.imshow("Image", img)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    print("nr of frames: ", nr)
    cap.release()
    # out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #play_check()
    read_txt()
