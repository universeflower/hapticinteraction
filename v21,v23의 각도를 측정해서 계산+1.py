import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# MediaPipe hands model 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def calculate_angle(point1, point2, point3):
    """세 점 사이의 각도 계산"""
    vector1 = np.array(point1) - np.array(point2)
    vector2 = np.array(point3) - np.array(point2)
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    cosine_theta = dot_product / (norm1 * norm2)
    angle = np.arccos(cosine_theta)
    angle_degrees = np.degrees(angle)
    return angle_degrees

# 이전 랜드마크 초기화
captured_landmarks = None

# 각도 저장을 위한 리스트
angles_21_23 = []
angles_32_34 = []
timestamps = []

# 웹캠 실행
cap = cv2.VideoCapture(0)

start_time = datetime.now()
end_time = start_time + timedelta(seconds=30)

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

            # 손 랜드마크를 넘파이 배열로 변환 (x, y, z 좌표 및 가시성 포함)
            landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in hand_landmarks.landmark])

            # 이전 랜드마크가 캡쳐되었는지 확인하고 캡쳐되지 않았으면 현재 랜드마크를 캡쳐
            if captured_landmarks is None:
                captured_landmarks = landmarks
                continue
            if cv2.waitKey(1) == ord('s'):
                print(angles_21_23)
                print(angles_32_34)
            # 현재 시간 저장
            current_time = datetime.now()

            # start_time과 end_time 사이에서만 각도 저장
            if start_time <= current_time <= end_time:
                # 0.5초 간격으로 저장
                if (current_time - start_time).total_seconds() % 0.5 == 0:
                    # v21번과 v23사이의 각, v32과 v34사이의 각을 계산하여 각 손가락의 구부러짐 정도를 측정
                    angle_21_23 = calculate_angle(captured_landmarks[1], captured_landmarks[2], captured_landmarks[3])
                    angle_32_34 = calculate_angle(captured_landmarks[2], captured_landmarks[3], captured_landmarks[4])
                    
                    # 각도와 타임스탬프 저장
                    angles_21_23.append(angle_21_23)
                    angles_32_34.append(angle_32_34)
                    timestamps.append((current_time - start_time).total_seconds())

    # 화면에 출력
    cv2.imshow('Hand Landmark Detection', img)

    # 종료 조건
    if cv2.waitKey(1) == ord('q') or datetime.now() >= end_time:
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()

# 각도 그래프 시각화
plt.plot(timestamps, angles_21_23, label='Angle between vectors 12-23')
plt.plot(timestamps, angles_32_34, label='Angle between vectors 23-34')
plt.xlabel('Time (seconds)')
plt.ylabel('Angle (degrees)')
plt.title('Finger Flexion Angles Over Time')
plt.legend()
plt.show()
