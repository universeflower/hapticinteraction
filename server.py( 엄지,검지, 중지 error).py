import socket
import numpy as np
import struct
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import time
import math
from datetime import datetime

# One Euro Filter 클래스 정의
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

# 서버 소켓 설정
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 65432))  # 모든 인터페이스에서 접속 허용
server_socket.listen(5)

print("Server is listening...")

cap = cv2.VideoCapture(0)

while True:
    client_socket, addr = server_socket.accept()
    print(f"Connected by {addr}")

    # One Euro Filter 초기화
    filter_1 = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    filter_2 = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    filter_3 = OneEuroFilter(min_cutoff=1.0, beta=0.0)

    angles_1 = []
    angles_2 = []
    angles_3 = []
    received_angles_1 = []
    received_angles_2 = []
    received_angles_3 = []
    errors_1 = []
    errors_2 = []
    errors_3 = []

    start_time = time.time()

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
                
                # 각도 계산
                angle_1 = calculate_angle(landmarks[1], landmarks[2], landmarks[3])
                angle_2 = calculate_angle(landmarks[5], landmarks[6], landmarks[7])
                angle_3 = calculate_angle(landmarks[9], landmarks[10], landmarks[11])
                
                # 현재 시간 계산
                t = time.time()

                # One Euro Filter 적용
                filtered_angle_1 = filter_1.update(t, angle_1)
                filtered_angle_2 = filter_2.update(t, angle_2)
                filtered_angle_3 = filter_3.update(t, angle_3)

                # 각도 데이터를 클라이언트로부터 수신
                data = client_socket.recv(12)
                if not data:
                    break

                received_angle_1, received_angle_2, received_angle_3 = struct.unpack('fff', data)
                
                # 오차 계산
                error_1 = abs(filtered_angle_1 - received_angle_1)
                error_2 = abs(filtered_angle_2 - received_angle_2)
                error_3 = abs(filtered_angle_3 - received_angle_3)

                # 데이터 저장
                angles_1.append(filtered_angle_1)
                angles_2.append(filtered_angle_2)
                angles_3.append(filtered_angle_3)
                received_angles_1.append(received_angle_1)
                received_angles_2.append(received_angle_2)
                received_angles_3.append(received_angle_3)
                errors_1.append(error_1)
                errors_2.append(error_2)
                errors_3.append(error_3)

                current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"Received Time: {current_time}, Filtered Angle 1: {filtered_angle_1}, Received Angle 1: {received_angle_1}, Error 1: {error_1}")
                print(f"Received Time: {current_time}, Filtered Angle 2: {filtered_angle_2}, Received Angle 2: {received_angle_2}, Error 2: {error_2}")
                print(f"Received Time: {current_time}, Filtered Angle 3: {filtered_angle_3}, Received Angle 3: {received_angle_3}, Error 3: {error_3}")

        # 20초가 지나면 루프 종료
        if time.time() - start_time > 20:
            break

        cv2.imshow('Hand Landmark Detection', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    client_socket.close()

    # 그래프 그리기
    plt.figure(figsize=(12, 18))

    plt.subplot(3, 3, 1)
    plt.plot(angles_1, label='Filtered Angle 1')
    plt.ylabel('Angle 1 (degrees)')
    plt.legend()

    plt.subplot(3, 3, 2)
    plt.plot(received_angles_1, label='Received Angle 1', color='orange')
    plt.ylabel('Angle 1 (degrees)')
    plt.legend()

    plt.subplot(3, 3, 3)
    plt.plot(errors_1, label='Error 1', color='red')
    plt.xlabel('Frame')
    plt.ylabel('Error 1 (degrees)')
    plt.legend()

    plt.subplot(3, 3, 4)
    plt.plot(angles_2, label='Filtered Angle 2')
    plt.ylabel('Angle 2 (degrees)')
    plt.legend()

    plt.subplot(3, 3, 5)
    plt.plot(received_angles_2, label='Received Angle 2', color='orange')
    plt.ylabel('Angle 2 (degrees)')
    plt.legend()

    plt.subplot(3, 3, 6)
    plt.plot(errors_2, label='Error 2', color='red')
    plt.xlabel('Frame')
    plt.ylabel('Error 2 (degrees)')
    plt.legend()

    plt.subplot(3, 3, 7)
    plt.plot(angles_3, label='Filtered Angle 3')
    plt.ylabel('Angle 3 (degrees)')
    plt.legend()

    plt.subplot(3, 3, 8)
    plt.plot(received_angles_3, label='Received Angle 3', color='orange')
    plt.ylabel('Angle 3 (degrees)')
    plt.legend()

    plt.subplot(3, 3, 9)
    plt.plot(errors_3, label='Error 3', color='red')
    plt.xlabel('Frame')
    plt.ylabel('Error 3 (degrees)')
    plt.legend()

    plt.tight_layout()
    plt.show()

cap.release()
cv2.destroyAllWindows()
