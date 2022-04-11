import cv2
import mediapipe as mp
import time
import xlsxwriter

# write to excel
book = xlsxwriter.Workbook('BP_f10.xlsx')
sheet = book.add_worksheet()

def detect():
    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()

    cap = cv2.VideoCapture('mat_color.mp4')
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
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 8, 8), 3)
            cv2.imshow("Image", img)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    book.close()

if __name__ == '__main__':
    detect()