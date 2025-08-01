import cv2
import numpy as np

def nothing(x):
    """트랙바 콜백 함수 - 아무것도 하지 않지만 필수로 필요"""
    pass

# 웹캠 시작
cap = cv2.VideoCapture(0)

# 트랙바를 위한 윈도우 생성
cv2.namedWindow('Controls')

# HSV 범위 조절을 위한 트랙바 생성
# Hue는 0-179 범위 (OpenCV에서는 0-179를 사용)
cv2.createTrackbar('H Min', 'Controls', 0, 179, nothing)
cv2.createTrackbar('H Max', 'Controls', 10, 179, nothing)
# Saturation과 Value는 0-255 범위
cv2.createTrackbar('S Min', 'Controls', 120, 255, nothing)
cv2.createTrackbar('S Max', 'Controls', 255, 255, nothing)
cv2.createTrackbar('V Min', 'Controls', 70, 255, nothing)
cv2.createTrackbar('V Max', 'Controls', 255, 255, nothing)

print("트랙바를 조절해서 원하는 색상을 찾아보세요")
print("빨간색 공을 찾으려면 H Min: 0, H Max: 10 정도로 시작해보세요")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 가우시안 블러로 노이즈 제거
    # 실제 환경에서는 조명이나 카메라 노이즈 때문에 깔끔한 결과를 얻기 어려움
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    
    # HSV로 변환
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # 트랙바에서 현재 값 읽기
    h_min = cv2.getTrackbarPos('H Min', 'Controls')
    h_max = cv2.getTrackbarPos('H Max', 'Controls')
    s_min = cv2.getTrackbarPos('S Min', 'Controls')
    s_max = cv2.getTrackbarPos('S Max', 'Controls')
    v_min = cv2.getTrackbarPos('V Min', 'Controls')
    v_max = cv2.getTrackbarPos('V Max', 'Controls')
    
    # HSV 범위 정의
    lower_range = np.array([h_min, s_min, v_min])
    upper_range = np.array([h_max, s_max, v_max])
    
    # 마스크 생성 - 지정한 범위 내의 색상만 흰색(255)으로, 나머지는 검은색(0)으로
    mask = cv2.inRange(hsv, lower_range, upper_range)
    
    # 모폴로지 연산으로 마스크 개선
    # 작은 구멍을 메우고 노이즈를 제거
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 열림 연산: 노이즈 제거
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 닫힘 연산: 구멍 메우기
    
    # 마스크를 원본 영상에 적용
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    # 결과를 2x2 그리드로 표시
    # 상단: 원본, HSV
    # 하단: 마스크, 결과
    top_row = np.hstack([frame, hsv])
    
    # 마스크는 1채널이므로 3채널로 변환해서 표시
    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    bottom_row = np.hstack([mask_3channel, result])
    
    # 전체 결과 조합
    display = np.vstack([top_row, bottom_row])
    
    # 크기 조정
    height, width = display.shape[:2]
    scale = 800 / width  # 너비를 800픽셀로 고정
    new_width = int(width * scale)
    new_height = int(height * scale)
    display = cv2.resize(display, (new_width, new_height))
    
    cv2.imshow('Color Detection Results', display)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()