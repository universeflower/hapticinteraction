import cv2
import mediapipe as mp
import numpy as np
import os
import time

# MediaPipe hands model 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# 웹캠 실행
cap = cv2.VideoCapture(0)

# 데이터 저장을 위한 빈 리스트
data = []

# 'r' 키를 누른 시간 저장
last_capture_time = 0

while cap.isOpened():
    ret, img = cap.read()

    # 이미지 좌우 반전
    img = cv2.flip(img, 1)

    # 이미지를 RGB로 변환
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 미디어 파이프로 손 감지
    results = hands.process(img_rgb)

    # 현재 시간 저장
    current_time = time.time()

    # 손이 감지되었을 때만 랜드마크 시각화 및 데이터 수집
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 손 랜드마크 시각화
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 손 랜드마크 좌표 추출
            joint = np.zeros((21, 4))
            for j, lm in enumerate(hand_landmarks.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # Compute angles between joints
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent joint
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]  # Child joint
            v = v2 - v1  # [20, 3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

            angle = np.degrees(angle)  # Convert radian to degree

            # 각도 데이터를 데이터 리스트에 추가
            data.append(angle)

            # 'r' 키를 누를 때마다 이미지 캡처 및 데이터 저장
            if cv2.waitKey(1) == ord('r') and current_time - last_capture_time > 1:
                # 이미지 캡처
                timestamp = int(time.time())
                cv2.imwrite(os.path.join('captured_images', f'captured_{timestamp}.jpg'), img)
                print(f'Image captured at {timestamp}')

                # 데이터 저장
                np.save(os.path.join('hand_landmark_data', f'hand_angles_{timestamp}.npy'), np.array(data))
                print(f'Hand angles data saved at {timestamp}')

                # 'r' 키를 누른 시간 갱신
                last_capture_time = current_time
    else:
        # 손이 감지되지 않은 경우에도 이미지 캡처
        if cv2.waitKey(1) == ord('r') and current_time - last_capture_time > 1:
            # 이미지 캡처
            timestamp = int(time.time())
            cv2.imwrite(os.path.join('captured_images', f'captured_{timestamp}.jpg'), img)
            print(f'Image captured at {timestamp}')

            # 데이터 저장
            np.save(os.path.join('hand_landmark_data', f'hand_angles_{timestamp}.npy'), np.array(data))
            print(f'Hand angles data saved at {timestamp}')

            # 'r' 키를 누른 시간 갱신
            last_capture_time = current_time

    # 화면에 출력
    cv2.imshow('Hand Landmark Detection', img)

    # 종료 조건
    if cv2.waitKey(1) == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
