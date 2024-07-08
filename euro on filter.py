import pandas as pd
import numpy as np

# CSV 파일 읽기
file_path = '/mnt/data/angles_index.csv'
data = pd.read_csv(file_path)

# 데이터 확인
print(data.head())

# One Euro Filter 클래스 구현
class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0, freq=120.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.freq = freq
        self.x_prev = None
        self.dx_prev = None
        self.first_time = True
    
    def alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)
    
    def filter(self, x):
        if self.first_time:
            self.x_prev = x
            self.dx_prev = 0.0
            self.first_time = False
        
        te = 1.0 / self.freq
        dx = (x - self.x_prev) / te
        edx = self.dx_prev + self.alpha(self.d_cutoff) * (dx - self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(edx)
        filtered_x = self.x_prev + self.alpha(cutoff) * (x - self.x_prev)
        
        self.x_prev = filtered_x
        self.dx_prev = edx
        
        return filtered_x

# 필터 초기화 (파라미터는 필요에 따라 조정할 수 있습니다)
min_cutoff = 1.0
beta = 0.0
d_cutoff = 1.0
freq = 120.0
one_euro_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff, freq=freq)

# 필터링 적용
filtered_data = data.copy()
for column in data.columns:
    filtered_data[column] = data[column].apply(one_euro_filter.filter)

# 결과 저장
filtered_file_path = '/mnt/data/filtered_angles_index.csv'
filtered_data.to_csv(filtered_file_path, index=False)

print(f"Filtered data saved to {filtered_file_path}")