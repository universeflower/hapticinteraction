import math

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def alpha(self, cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, t, x):
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        dt = t - self.t_prev
        dx = (x - self.x_prev) / dt

        alpha_d = self.alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self.alpha(cutoff, dt)
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev

        self.t_prev = t
        self.x_prev = x_hat
        self.dx_prev = dx_hat

        return x_hat
import socket
import struct
import mediapipe as mp
import cv2
import numpy as np
import time
from datetime import datetime

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

# 클라이언트 소켓 설정
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('172.21.37.148', 65432))  # 여기서 '서버_로컬_IP_주소'를 서버 컴퓨터의 로컬 IP 주소로 변경

# One Euro Filter 초기화
filter = OneEuroFilter(min_cutoff=1.0, beta=0.0)
s_t = time.time()
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        break

    # 이미지 좌우 반전 및 RGB 변환
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 미디어파이프로 손 감지
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            
            # 각도 계산 (1, 2, 3번 랜드마크 사용)
            angle_21_23 = calculate_angle(landmarks[1], landmarks[2], landmarks[3])
            
            # 현재 시간 계산
            t = time.time()

            # One Euro Filter 적용
            filtered_angle = filter.update(t, angle_21_23)
            
            elapsed_time = t - s_t
            if elapsed_time > 10:
                data = struct.pack('f', filtered_angle)
                client_socket.send(data)
                timestamp = datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]
                print(f"Sent Angle: {filtered_angle}")
                print(timestamp)
                
                # 화면에 메시지 표시
                cv2.putText(img, f"Go -current time :{elapsed_time}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                remaining_time = 10 - int(elapsed_time)
                cv2.putText(img, f"wait 10sec - remaining time: {remaining_time}sec", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
    
    
    cv2.imshow('Hand Landmark Detection - Client', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

client_socket.close()
cap.release()
cv2.destroyAllWindows()
