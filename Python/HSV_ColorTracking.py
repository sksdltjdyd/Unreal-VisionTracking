import cv2
import numpy as np
from pythonosc import udp_client

# 각 색상의 HSV 범위를 정의합니다
# 이 값들은 실험을 통해 찾은 일반적인 값이에요. 조명에 따라 조정이 필요할 수 있습니다.
COLOR_RANGES = {
    'red': {
        'lower': [(0, 120, 70), (170, 120, 70)],  # 빨간색은 0도와 180도 근처에 있어서 두 범위 필요
        'upper': [(4, 255, 255), (180, 255, 255)],
        'rgb': (0, 0, 255)  # 표시용 색상 (BGR 형식)
    },
    'blue': {
        'lower': [(100, 120, 70)],  # 파란색은 하나의 범위로 충분
        'upper': [(130, 255, 255)],
        'rgb': (255, 0, 0)
    },
    'green': {
        'lower': [(40, 50, 50)],
        'upper': [(80, 255, 255)],
        'rgb': (0, 255, 0)
    },
    'yellow': {
        'lower': [(20, 100, 100)],
        'upper': [(30, 255, 255)],
        'rgb': (0, 255, 255)
    },
    'orange': {
        'lower': [(10, 100, 100)],
        'upper': [(20, 255, 255)],
        'rgb': (0, 165, 255)
    },
    'purple': {
        'lower': [(130, 50, 50)],
        'upper': [(170, 255, 255)],
        'rgb': (255, 0, 255)
    }
}

def create_color_mask(hsv_image, color_name):
    """
    특정 색상에 대한 마스크를 생성하는 함수
    여러 범위를 가진 색상(예: 빨간색)도 처리할 수 있습니다
    """
    color_info = COLOR_RANGES[color_name]
    mask = None
    
    # 각 색상 범위에 대해 마스크 생성
    for i in range(len(color_info['lower'])):
        lower = np.array(color_info['lower'][i])
        upper = np.array(color_info['upper'][i])
        
        # 현재 범위에 대한 마스크 생성
        current_mask = cv2.inRange(hsv_image, lower, upper)
        
        # 첫 번째 마스크이면 그대로 사용, 아니면 OR 연산으로 합치기
        if mask is None:
            mask = current_mask
        else:
            mask = cv2.bitwise_or(mask, current_mask)
    
    return mask

# 사용자가 추적할 색상을 선택할 수 있게 합니다
print("추적할 색상을 선택하세요:")
print("사용 가능한 색상:", list(COLOR_RANGES.keys()))
selected_color = input("색상 이름을 입력하세요 (기본값: blue): ").lower()

if selected_color not in COLOR_RANGES:
    selected_color = 'blue'
    print(f"잘못된 색상입니다. {selected_color}를 사용합니다.")

cap = cv2.VideoCapture(0)
osc_client = udp_client.SimpleUDPClient("127.0.0.1", 5005)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 전처리
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # 선택한 색상에 대한 마스크 생성
    mask = create_color_mask(hsv, selected_color)
    
    # 모폴로지 연산으로 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > 500:
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # 색상별로 다른 OSC 주소 사용
                osc_address = f"/tracking/{selected_color}/position"
                norm_x = cx / frame.shape[1]
                norm_y = cy / frame.shape[0]
                osc_client.send_message(osc_address, [norm_x, norm_y])
                
                # 시각화
                cv2.circle(frame, (cx, cy), 10, COLOR_RANGES[selected_color]['rgb'], -1)
                cv2.putText(frame, f"{selected_color}: ({cx}, {cy})", 
                           (cx + 15, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, COLOR_RANGES[selected_color]['rgb'], 2)
    
    # 현재 추적 중인 색상 표시
    cv2.putText(frame, f"Tracking: {selected_color}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, COLOR_RANGES[selected_color]['rgb'], 2)
    
    cv2.imshow('Color Tracking', frame)
    cv2.imshow('Mask', mask)
    
    # 다른 색상으로 변경할 수 있게 키 매핑
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == ord('r'):
        selected_color = 'red'
    elif key == ord('b'):
        selected_color = 'blue'
    elif key == ord('g'):
        selected_color = 'green'
    elif key == ord('y'):
        selected_color = 'yellow'

cap.release()
cv2.destroyAllWindows()