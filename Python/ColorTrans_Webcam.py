import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # BGR을 HSV로 변환 (OpenCV는 RGB가 아닌 BGR 순서를 사용)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 원본과 HSV 영상을 나란히 표시해서 차이를 볼 수 있게 합니다
    # numpy의 hstack은 horizontal stack의 약자로, 가로로 이미지를 붙임
    combined = np.hstack([frame, hsv])
    
    # 창 크기를 조정해서 보기 편하게 만들기
    # 원본 크기가 너무 크면 화면에 다 안 보일 수 있다
    height, width = combined.shape[:2]
    if width > 1200:
        scale = 1200 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        combined = cv2.resize(combined, (new_width, new_height))
    
    cv2.imshow('Original (left) vs HSV (right)', combined)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()