import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# 빨간색 범위 정의 (실험을 통해 찾은 값)
# 빨간색은 HSV에서 0도와 180도 근처에 있어서 두 범위를 체크
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([4, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 전처리
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # 빨간색 마스크 생성 (두 범위를 OR 연산으로 합침)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # 모폴로지 연산
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 컨투어 찾기
    # RETR_EXTERNAL: 가장 바깥쪽 컨투어만 찾기
    # CHAIN_APPROX_SIMPLE: 컨투어를 간단하게 표현
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 화면에 정보 표시할 준비
    info_display = frame.copy()
    
    if contours:
        # 가장 큰 컨투어 찾기 (가장 큰 빨간색 객체)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # 너무 작은 객체는 무시 (노이즈 제거)
        if area > 500:
            # 모멘트 계산으로 중심점 찾기
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])  # x 중심
                cy = int(M["m01"] / M["m00"])  # y 중심
                
                # 객체를 감싸는 원 그리기
                ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
                
                if radius > 10:
                    # 중심점에 작은 원 그리기
                    cv2.circle(info_display, (cx, cy), 5, (0, 255, 0), -1)
                    # 객체를 감싸는 큰 원 그리기
                    cv2.circle(info_display, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    
                    # 정보 텍스트 추가
                    cv2.putText(info_display, f"Center: ({cx}, {cy})", 
                               (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (0, 255, 0), 2)
                    cv2.putText(info_display, f"Area: {int(area)}", 
                               (cx + 10, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (0, 255, 0), 2)
                    
                    # 화면 좌표를 정규화된 좌표로 변환 (0~1 범위)
                    norm_x = cx / frame.shape[1]
                    norm_y = cy / frame.shape[0]
                    
                    # 상단에 정규화된 좌표 표시
                    cv2.putText(info_display, 
                               f"Normalized: ({norm_x:.3f}, {norm_y:.3f})", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 255, 255), 2)
    
    # 결과 표시
    cv2.imshow('Object Detection', info_display)
    cv2.imshow('Mask', mask)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()