import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480
# wCam, hCam = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)


folderPath = "NumberImage"
myList = os.listdir(folderPath)
# print(myList)
overlaylist = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlaylist.append(image)

# print(len(overlaylist))
pTime = 0

detector = htm.handDetector(max_hands=1, detection_confidence=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)
    if len(lmlist) != 0:
        fingers = []

        # Thumb
        if lmlist[tipIds[0]][1] > lmlist[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers(也可以对单独手指进行判断，形成 01011 二进制来代表某一种情况)
        for id in range(1, 5):
            if lmlist[tipIds[id]][2] < lmlist[tipIds[id]-2][2]:
                # print('Index finger open')
                fingers.append(1)
            else:
                fingers.append(0)
            # print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)

        h, w, c = overlaylist[totalFingers].shape
        img[:h, :w] = overlaylist[totalFingers]

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)


    cv2.imshow('Image', img)
    cv2.waitKey(1)