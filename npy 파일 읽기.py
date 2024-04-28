import numpy as np

# 읽어올 npy 파일 경로
file_path = "dataset\seq_left_1712312496.npy"

# npy 파일 읽기
data = np.load(file_path)

# 데이터 출력
print("손 랜드마크 데이터:")
print(data)
