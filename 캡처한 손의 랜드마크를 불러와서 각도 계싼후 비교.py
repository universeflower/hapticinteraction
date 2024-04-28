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

# 웹캠 실행
cap = cv2.VideoCapture(0)

# 캡쳐한 손의 랜드마크 불러오기
captured_landmarks = np.load('hand_landmark_data\hand_landmarks_1712937902.npy')

# 손목에서 엄지손가락까지의 벡터를 계산하는 함수
def calculate_wrist_to_thumb_vector(landmarks):
    wrist = landmarks[0]
    thumb = landmarks[1]
    return thumb - wrist

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

            # 캡처한 손의 랜드마크와 현재 손의 랜드마크 간의 오차 계산
            error = np.linalg.norm(captured_landmarks - landmarks)

            # 손목에서 엄지손가락까지의 벡터 계산
            vector_wrist_to_thumb = calculate_wrist_to_thumb_vector(landmarks)

            # 캡처한 손의 손목에서 엄지손가락까지의 벡터 계산
            captured_vector_wrist_to_thumb = calculate_wrist_to_thumb_vector(captured_landmarks)

            # 내적 계산
            dot_product = np.dot(vector_wrist_to_thumb, captured_vector_wrist_to_thumb)

            # 내적이 0 이상인 경우 "correct" 출력
            if dot_product >= 0:
                print("correct")

    # 화면에 출력
    cv2.imshow('Hand Landmark Detection', img)

    # 종료 조건
    if cv2.waitKey(1) == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()

