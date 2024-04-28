import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# 두 개의 npy 파일에서 손의 landmark 좌표를 읽어옵니다.
hand1_landmarks = np.load('hand_landmark_data/hand_landmarks_1712937902.npy')
hand2_landmarks = np.load('hand_landmark_data/hand_landmarks_1712937895.npy')

error = hand2_landmarks - hand1_landmarks

# 각 좌표 간의 거리를 계산하여 scalar 값을 얻습니다.
error_magnitude = np.linalg.norm(error, axis=1)

best_similarity = 0.0
best_params = {}

# PID 제어기를 설정합니다.
for kp in np.linspace(0.01, 2, num=20):  # 비례 게인을 0.01부터 2까지 20개로 조정합니다.
    for ki in np.linspace(0.001, 0.5, num=20):  # 적분 게인을 0.001부터 0.5까지 20개로 조정합니다.
        for kd in np.linspace(0.001, 0.5, num=20):  # 미분 게인을 0.001부터 0.5까지 20개로 조정합니다.
            # 각 손의 랜드마크 좌표에 대해 PID 제어기를 적용합니다.
            corrected_hand2_landmarks = np.zeros(hand2_landmarks.shape)
            cumulative_error = np.zeros_like(error_magnitude)
            for i in range(hand2_landmarks.shape[0]):
                # 현재 오차를 기반으로 PID 제어를 수행합니다.
                current_error = error_magnitude[i]

                # PID 제어 알고리즘을 적용합니다.
                pid_output = kp * current_error + ki * np.sum(cumulative_error) + kd * (current_error - cumulative_error[i])

                # 보정된 오차를 사용하여 두 번째 손의 landmark 좌표를 수정합니다.
                corrected_hand2_landmarks[i] = hand2_landmarks[i] - (pid_output * (error[i] / np.linalg.norm(error[i])))

                # 누적 오차를 업데이트합니다.
                cumulative_error[i] += current_error

            # 유클리드 거리를 사용하여 두 개의 landmark 좌표 간의 유사도를 계산합니다.
            similarity = np.mean(1 / (1 + euclidean_distances(hand1_landmarks, corrected_hand2_landmarks)))  # 유사도를 0과 1 사이의 값으로 제한

            # 가장 높은 유사도와 해당하는 게인들을 저장합니다.
            if similarity > best_similarity:
                best_similarity = similarity
                best_params = {'kp': kp, 'ki': ki, 'kd': kd}

print(f"가장 높은 유사도: {best_similarity}, 해당하는 게인들: {best_params}")
