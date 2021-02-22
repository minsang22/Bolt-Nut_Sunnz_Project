## -*- coding: utf-8 -*-  # 한글 주석쓸려면 적기

import cv2

font = cv2.FONT_ITALIC


def faceDetect():
    mouth_detect = True
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")  # 얼굴찾기 haar 파일
    mouth_cascade = cv2.CascadeClassifier("haarcascade_mouth.xml")  # 입찾기 haar 파일
    sideface_cascade = cv2.CascadeClassifier("haarcascade_profileface.xml") #측면얼굴찾기 haar 파

    try:
        #cam = cv2.VideoCapture("src.mp4")
        cam = cv2.VideoCapture(0)
    except:
        print("camera loading error")
        return

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        if mouth_detect:
            info = "Mouth Detention ON"
        else:
            info = "Mouth Detection OFF"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(20,20))
        #faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # 카메라 영상 왼쪽위에 위에 셋팅된 info 의 내용 출력
        cv2.putText(frame, info, (5, 15),  font, 0.5, (255, 0, 255), 1)

        if (len(faces) != 0):
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 사각형 범위
                cv2.putText(frame, "Detected Face", (x - 5, y - 5), font, 0.5, (255, 255, 0), 2)  # 얼굴찾았다는 메시지
                if mouth_detect:  # 눈찾기
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = frame[y:y + h, x:x + w]
                    mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=30)
                    for (ex, ey, ew, eh) in mouths:
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        elif (len(faces) == 0):
            sidefaces = sideface_cascade.detectMultiScale(gray, 1.2, 5, minSize=(15,15))

            for (x, y, w, h) in sidefaces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 사각형 범위
                cv2.putText(frame, "Detected SideFace", (x - 5, y - 5), font, 0.5, (255, 255, 0), 2)  # 얼굴찾았다는 메시지
                if mouth_detect:  # 눈찾기
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_color = frame[y:y + h, x:x + w]
                    mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=30)
                    for (ex, ey, ew, eh) in mouths:
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("frame", frame)

        #k = cv2.waitKey(30)

        # 실행 중 키보드 i 를 누르면 눈찾기를 on, off한다.
        #if k == ord('i'):
            #mouth_detect = not mouth_detect

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()


faceDetect()