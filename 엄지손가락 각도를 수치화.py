import cv2
import mediapipe as mp
import numpy as np

# MediaPipe hands model 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# 이전 랜드마크 초기화
captured_landmarks = None

# 웹캠 실행
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()

    # 이미지 좌우 반전
    img = cv2.flip(img, 1)

    # 이미지를 RGB로 변환
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 미디어 파이프로 손 감지
    results = hands.process(img_rgb)

    # 손이 감지되었을 때만 랜드마크 시각화 및 오차 계산
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 손 랜드마크 시각화
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 손 랜드마크를 넘파이 배열로 변환 (x, y 좌표만 사용)
            landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])

            # 이전 랜드마크가 캡쳐되었는지 확인하고 캡쳐되지 않았으면 현재 랜드마크를 캡쳐
            if captured_landmarks is None:
                captured_landmarks = landmarks
                continue
            
            # 12, 23, 34번 좌표를 기준으로 각도 계산
            point1 = captured_landmarks[1]
            point2 = captured_landmarks[2]
            point3 = captured_landmarks[3]
            angle_12_23 = np.degrees(np.arctan2(point3[1] - point2[1], point3[0] - point2[0]) - np.arctan2(point1[1] - point2[1], point1[0] - point2[0]))

            point1 = captured_landmarks[2]
            point2 = captured_landmarks[3]
            point3 = captured_landmarks[4]
            angle_23_34 = np.degrees(np.arctan2(point3[1] - point2[1], point3[0] - point2[0]) - np.arctan2(point1[1] - point2[1], point1[0] - point2[0]))

            # 각도 출력
            if cv2.waitKey(1) == ord('s'):
             print("Angle 12-23:", angle_12_23)
             print("Angle 23-34:", angle_23_34)

            # 엄지 손가락이 접혀있는지 확인
            if angle_12_23 <= 90 or angle_23_34  <=90:
                print("Thumb folded")

            # 현재 랜드마크를 이전 랜드마크로 업데이트
            captured_landmarks = landmarks

    # 화면에 출력
    cv2.imshow('Hand Landmark Detection', img)

    # 종료 조건
    if cv2.waitKey(1) == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
