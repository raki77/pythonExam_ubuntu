import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# 감정 분류를 위한 클래스 레이블
emotion_labels = ['Angry', 'Disgust', 'Fear',
                  'Happy', 'Sad', 'Surprise', 'Neutral']

# 사전 학습된 얼굴 인식 모델 로드 (OpenCV 제공)
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 사전 학습된 감정 인식 모델 로드
# emotion_classifier = load_model('emotion_model.hdf5')
emotion_classifier = load_model('emotion_model.hdf5', compile=False)

# 실시간 웹캠 피드 시작
cap = cv2.VideoCapture(0)

while True:
    # 웹캠으로부터 프레임을 읽기
    ret, frame = cap.read()
    labels = []

    # 흑백 이미지로 변환
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 인식
    faces = face_classifier.detectMultiScale(
        gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # 얼굴 영역을 ROI로 추출
        roi_gray = gray_frame[y:y+h, x:x+w]
        # roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)

        # 이미지 전처리
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # 감정 예측
            prediction = emotion_classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y - 10)

            # 결과 화면에 표시
            cv2.putText(frame, label, label_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 화면 출력
    cv2.imshow('Emotion Detector', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 종료 및 모든 창 닫기
cap.release()
cv2.destroyAllWindows()
