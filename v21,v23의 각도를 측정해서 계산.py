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

# 각도 계산 함수
def calculate_angle(point1, point2, point3):
    """세 점 사이의 각도 계산"""
    radians = np.arctan2(point3[1] - point2[1], point3[0] - point2[0]) - np.arctan2(point1[1] - point2[1], point1[0] - point2[0])
    angle = np.abs(np.degrees(radians))
    return angle

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

            # 손 랜드마크를 넘파이 배열로 변환 (x, y, z 좌표 및 가시성 포함)
            landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in hand_landmarks.landmark])

            # 이전 랜드마크가 캡쳐되었는지 확인하고 캡쳐되지 않았으면 현재 랜드마크를 캡쳐
            if captured_landmarks is None:
                captured_landmarks = landmarks
                continue
            
            # v21과 v23 사이의 각, v32와 v34 사이의 각을 계산하여 각 손가락의 구부러짐 정도를 측정
            angle_21_23 = calculate_angle(captured_landmarks[0], captured_landmarks[1], captured_landmarks[2])
            angle_32_34 = calculate_angle(captured_landmarks[1], captured_landmarks[2], captured_landmarks[3])

            # 구부러짐 각이 90도 미만이면 엄지 손가락이 접혀있다고 판단
            if angle_21_23 < 90 and angle_32_34 < 90:
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
