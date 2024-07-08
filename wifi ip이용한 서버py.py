import socket
import numpy as np
import struct
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
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

# 서버 소켓 설정
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 65432))  # 모든 인터페이스에서 접속 허용
server_socket.listen(5)

print("Server is listening...")

cap = cv2.VideoCapture(0)

while True:
    client_socket, addr = server_socket.accept()
    print(f"Connected by {addr}")

    angles = []
    received_angles = []
    errors = []

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
                
                # 각도 계산 (1, 2, 3번 랜드마크 사용)
                #angle_21_23 = calculate_angle(landmarks[1], landmarks[2], landmarks[3])
                angle_65_67 = calculate_angle(landmarks[5], landmarks[6], landmarks[7])
                #angle_109_1011 = calculate_angle(landmarks[9], landmarks[10], landmarks[11])
                # 각도 데이터를 클라이언트로부터 수신
                data = client_socket.recv(4)
                if not data:
                    break

                received_angle = struct.unpack('f', data)[0]
                
                # 오차 계산
                error = abs(angle_65_67 - received_angle)

                # 데이터 저장
                angles.append(angle_65_67)
                received_angles.append(received_angle)
                errors.append(error)

                current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"Received Time: {current_time}, Received Angle: {received_angle}, Current Angle: {angle_65_67}, Error: {error}")

        # 10초가 지나면 루프 종료
        if time.time() - start_time > 20:
            break

        cv2.imshow('Hand Landmark Detection', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    client_socket.close()

    # 그래프 그리기
    plt.figure(figsize=(12, 6))

    plt.subplot(3, 1, 1)
    plt.plot(angles, label='Current Angle')
    plt.ylabel('Angle (degrees)')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(received_angles, label='Received Angle', color='orange')
    plt.ylabel('Angle (degrees)')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(errors, label='Error', color='red')
    plt.xlabel('Frame')
    plt.ylabel('Error (degrees)')
    plt.legend()

    plt.tight_layout()
    plt.show()

cap.release()
cv2.destroyAllWindows()
