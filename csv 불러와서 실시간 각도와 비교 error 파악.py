import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import csv

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

# CSV 파일에서 각도 데이터를 불러오기
angles_from_csv = []
with open('angles_thumb.csv', mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # 헤더 건너뛰기
    for row in reader:
        angles_from_csv.append(float(row[1]))

# 이전 랜드마크 초기화
captured_landmarks = None
real_time_angles = []
error_values = []

# 웹캠 실행
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

start_time = datetime.now()
end_time = start_time + timedelta(seconds=10)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        print("웹캠에서 프레임을 읽을 수 없습니다.")
        break

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
            
            # 각 손가락에 대한 랜드마크 각도 계산
            angle_21_23 = calculate_angle(captured_landmarks[1], captured_landmarks[2], captured_landmarks[3])
            
            real_time_angles.append(angle_21_23)
            if len(angles_from_csv) > len(real_time_angles):
                error = angle_21_23 - angles_from_csv[len(real_time_angles) - 1]
                error_values.append(error)
            else:
                error_values.append(0)

            captured_landmarks = landmarks

    # 화면에 출력
    cv2.imshow('Hand Landmark Detection', img)

    # 종료 조건
    if datetime.now() > end_time or cv2.waitKey(1) == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()

# 캡처된 각도를 그래프로 표시
plt.figure(figsize=(10, 6))
plt.plot(real_time_angles, label='Real-time Angle')
plt.plot(error_values, label='Error Value')
plt.xlabel('Frame')
plt.ylabel('Angle (degrees)')
plt.title('Real-time Angle and Error Value')
plt.legend()
plt.show()














