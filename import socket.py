import socket
import struct
import csv
import time

# 클라이언트 소켓 설정
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('localhost', 65432))

# CSV 파일 읽기
with open('C:\\Users\\user\\Desktop\\공압식제어\\angles_thumb.csv', mode='r') as file:
    csv_reader = csv.DictReader(file)
    
    for row in csv_reader:
        frame = int(row['Frame'])
        angle_thumb = float(row['Angle_Thumb'])
        
        # 각도 데이터를 바이트로 변환하여 서버로 전송
        data = struct.pack('f', angle_thumb)
        client_socket.send(data)
        print(f"Sent Frame {frame}: Angle_Thumb {angle_thumb}")
        
        # 30프레임 단위로 전송하기 위해 1초 대기
        time.sleep(1 / 30)

client_socket.close()
