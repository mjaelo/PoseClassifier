import cv2
import mediapipe as mp
import time
import xlsxwriter

# write to excel
vid='videos/f2_normal.mp4'
excel='BP/BP_f20.xlsx'
book = xlsxwriter.Workbook(excel)
sheet = book.add_worksheet()

# oraz pomoże to w zrozumieniu znaczenia warunków podczas nagrywania filmów do tych algorytmów

def detect():
    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    cap = cv2.VideoCapture(vid)
    pTime = 0

    row = 0
    column = 0
    while column <= 32:
        sheet.write(row, column, column)
        column += 1
    column = 0
    while cap.isOpened():
        success, img = cap.read()
        if success:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)

            # print(results.pose_landmarks)
            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = img.shape
                    print(id, lm)
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    # write to excel
                    if id == 0:
                        column = 0
                        row += 1
                    content = str([cx, cy])
                    sheet.write(row, column, content)
                    column += 1

                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
            else:
                column = 0
                row += 1
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 8, 8), 3)

            # resize image
            width = int(img.shape[1] / 2)
            height = int(img.shape[0] / 2)
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

            cv2.imshow("Image", img)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    column = 0
    row += 1
    sheet.write(row, column, 'endfile') # to prevent cell empty error at the end of file
    book.close()

if __name__ == '__main__':
    detect()