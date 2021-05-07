import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

def main():

    pTime = 0
    cTime = 0
    # cap = cv2.VideoCapture(0)
    # 调用ip摄像头
    video = "http://admin:admin@192.168.1.108:8081/"
    cap = cv2.VideoCapture(video)
    detector = htm.handDectector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            print(lmList[4])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, 'FPS:' + str(round(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 0, 255), 3)


        cv2.imshow("Image", img)
        cv2.waitKey(1)





if __name__ == '__main__':
    main()