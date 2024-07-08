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

def mean_normalization_angle(point1,point2):
    # 각도의 평균 계산
    avg_angle = (point1 + point2) / 2.0
    
    # 최소 및 최대 각도 지정
    min_angle = 105
    max_angle = 180
    
    # 정규화
    normalized_angle = (avg_angle - min_angle) / (max_angle - min_angle)
    
    # 값이 0보다 작으면 0으로, 1보다 크면 1로 설정
    normalized_angle = max(0, min(normalized_angle, 1))
    
    return normalized_angle



# 이전 랜드마크 초기화
captured_landmarks = None
angles_21_23 = []
angles_32_34 = []
angles_65_67 = []
angles_76_78 = []
angles_109_1011 = []
angles_1110_1112 = []
angles_1413_1415 = []
angles_1514_1516 = []
angles_1817_1819 = []
angles_1918_1920 = []
# 웹캠 실행
# video_file="C:/Users/user/Desktop/KakaoTalk_20240520_182109245.mp4"
cap = cv2.VideoCapture(0)

start_time = datetime.now()
end_time = start_time + timedelta(seconds=10)

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
            
            # 각 손가락에 대한 모든 랜드마크 각도 계산
            angle_21_23 = calculate_angle(captured_landmarks[1], captured_landmarks[2], captured_landmarks[3])
            angle_32_34 = calculate_angle(captured_landmarks[2], captured_landmarks[3], captured_landmarks[4])
            angle_65_67 = calculate_angle(captured_landmarks[5], captured_landmarks[6], captured_landmarks[7])
            angle_76_78 = calculate_angle(captured_landmarks[6], captured_landmarks[7], captured_landmarks[8])
            angle_109_1011 = calculate_angle(captured_landmarks[9], captured_landmarks[10], captured_landmarks[11])
            angle_1110_1112 = calculate_angle(captured_landmarks[10], captured_landmarks[11], captured_landmarks[12])
            angle_1413_1415 = calculate_angle(captured_landmarks[13], captured_landmarks[14], captured_landmarks[15])
            angle_1514_1516 = calculate_angle(captured_landmarks[14], captured_landmarks[15], captured_landmarks[16])
            angle_1817_1819 = calculate_angle(captured_landmarks[17], captured_landmarks[18], captured_landmarks[19])
            angle_1918_1920 = calculate_angle(captured_landmarks[18], captured_landmarks[19], captured_landmarks[20])

            angles_21_23.append((angle_21_23))
            angles_65_67.append((angle_65_67))
            angles_109_1011.append(angle_109_1011)
            # angles_1413_1415.append(angle_1413_1415)
            # angles_1817_1819.append(angle_1817_1819)
            # angles_21_23.append((angle_21_23+angle_32_34)/2)
            # angles_65_67.append((angle_65_67))
            # angles_109_1011.append((angle_76_78+angle_65_67)/2)
            # angles_1413_1415.append(mean_normalization_angle(angle_1413_1415,angle_1514_1516))
            # angles_1817_1819.append(mean_normalization_angle(angle_1817_1819,angle_1918_1920))
            captured_landmarks = landmarks

    # 화면에 출력()
    cv2.imshow('Hand Landmark Detection', img)

    # 0.5초마다 시간 체크 및 종료
    if datetime.now() > end_time:
        break

    # 종료 조건
    if cv2.waitKey(1) == ord('q'):
        break

#리소스 해제
cap.release()
cv2.destroyAllWindows()
with open('angles_thumb.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Frame', 'Angle_Thumb'])
    for i, angle in enumerate(angles_65_67):
        writer.writerow([i, angle])

# CSV 파일로 데이터 저장
# with open('hand_data.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Frame', 'Angle_Thumb', 'Angle_Index', 'Angle_Middle', 'Angle_Ring', 'Angle_Pinky'])
#     for i in range(len(angles_thumb)):
#         writer.writerow([i, angles_thumb[i], angles_index[i], angles_middle[i], angles_ring[i], angles_pinky[i]])

# 캡처된 각도를 그래프로 표시
plt.plot(angles_21_23, label='Angle between vectors thumb')
plt.plot( angles_109_1011, label='Angle between vectors middle')
plt.plot(angles_65_67, label='Angle between vectors index')
plt.plot(angles_1413_1415, label='Angle between vectors ring')
plt.plot(angles_1817_1819, label='Angle between vectors pinky')


plt.xlabel('Frame')
plt.ylabel('Angle (degrees)')
plt.title('Angle over time')
plt.legend()
plt.show()
