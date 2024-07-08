import socket
import numpy as np
import struct
import mediapipe as mp
import cv2

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
server_socket.bind(('localhost', 65432))
server_socket.listen(5)

print("Server is listening...")

cap = cv2.VideoCapture(0)

while True:
    client_socket, addr = server_socket.accept()
    print(f"Connected by {addr}")

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
                
                # 각도 데이터를 클라이언트로부터 수신
                data = client_socket.recv(4)
                if not data:
                    break

                received_angle = struct.unpack('f', data)[0]
                
                # 오차 계산
                error = abs(angle_21_23 - received_angle)
                print(f"Received Angle: {received_angle}, Current Angle: {angle_21_23}, Error: {error}")

        cv2.imshow('Hand Landmark Detection', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    client_socket.close()

cap.release()
cv2.destroyAllWindows()
