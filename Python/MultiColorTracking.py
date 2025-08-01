import cv2
import numpy as np
from pythonosc import udp_client
import time

# 각 색상의 HSV 범위를 정의
# 조명에 따라 조정이 필요
COLOR_RANGES = {
    'red': {
        'lower': [(0, 120, 70), (170, 120, 70)],  # 빨간색은 0도와 180도 근처에 있어서 두 범위 필요
        'upper': [(1, 255, 255), (180, 255, 255)],
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

class MultiColorTracker:
    """
    여러 색상을 동시에 추적하는 클래스
    각 색상의 상태를 독립적으로 관리
    """

    
    
    def __init__(self, colors_to_track=['red', 'blue', 'yellow']):
        self.colors_to_track = colors_to_track
        self.color_ranges = COLOR_RANGES
        self.osc_client = udp_client.SimpleUDPClient("127.0.0.1", 5005)
        
        # 각 색상별 추적 상태를 저장하는 딕셔너리
        self.tracking_data = {}
        for color in colors_to_track:
            self.tracking_data[color] = {
                'position': (0, 0),
                'detected': False,
                'last_seen': time.time(),
                'smooth_x': 0.5,
                'smooth_y': 0.5
            }


    def process_frame(self, frame):
        """한 프레임에서 모든 색상을 처리합니다"""
        # 전처리는 한 번만 수행 (효율성을 위해)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # 결과 표시용 이미지
        display = frame.copy()
        
        # 각 색상에 대해 처리
        for color in self.colors_to_track:
            # 마스크 생성
            mask = create_color_mask(hsv, color)
            
            # 모폴로지 연산
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 가장 큰 컨투어 선택
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > 300:  # 임계값을 좀 낮춰서 여러 객체를 잘 감지하도록
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # 추적 데이터 업데이트
                        self.tracking_data[color]['position'] = (cx, cy)
                        self.tracking_data[color]['detected'] = True
                        self.tracking_data[color]['last_seen'] = time.time()
                        
                        # 정규화된 좌표
                        norm_x = cx / frame.shape[1]
                        norm_y = cy / frame.shape[0]
                        
                        # 부드러운 움직임을 위한 보간
                        smooth_factor = 0.3
                        self.tracking_data[color]['smooth_x'] += (norm_x - self.tracking_data[color]['smooth_x']) * smooth_factor
                        self.tracking_data[color]['smooth_y'] += (norm_y - self.tracking_data[color]['smooth_y']) * smooth_factor
                        
                        # OSC 전송
                        self.osc_client.send_message(
                            f"/tracking/{color}/position", 
                            [self.tracking_data[color]['smooth_x'], 
                             self.tracking_data[color]['smooth_y']]
                        )
                        
                        # 시각화
                        self.visualize_detection(display, color, cx, cy, area)
                else:
                    self.tracking_data[color]['detected'] = False
            else:
                self.tracking_data[color]['detected'] = False
            
            # 일정 시간 이상 감지되지 않으면 'lost' 메시지 전송
            if not self.tracking_data[color]['detected']:
                time_since_seen = time.time() - self.tracking_data[color]['last_seen']
                if time_since_seen > 1.0:  # 1초 이상 안 보이면
                    self.osc_client.send_message(f"/tracking/{color}/lost", [1])
        
        # 상태 정보 표시
        self.draw_status(display)
        
        return display
    
    def visualize_detection(self, image, color, cx, cy, area):
        """검출된 객체를 시각화합니다"""
        color_bgr = self.color_ranges[color]['rgb']
        
        # 중심점에 원 그리기
        cv2.circle(image, (cx, cy), 10, color_bgr, -1)
        
        # 외곽선 그리기 (크기에 비례하는 원)
        radius = int(np.sqrt(area / np.pi))
        cv2.circle(image, (cx, cy), radius, color_bgr, 2)
        
        # 레이블 추가
        label = f"{color}: ({cx}, {cy})"
        cv2.putText(image, label, (cx + 15, cy - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
    
    def draw_status(self, image):
        """현재 추적 상태를 화면에 표시합니다"""
        y_offset = 30
        for i, color in enumerate(self.colors_to_track):
            status = "Active" if self.tracking_data[color]['detected'] else "Lost"
            color_bgr = self.color_ranges[color]['rgb']
            
            # 상태 표시
            text = f"{color}: {status}"
            cv2.putText(image, text, (10, y_offset + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
            
            # 상태 인디케이터 (원)
            indicator_color = color_bgr if self.tracking_data[color]['detected'] else (100, 100, 100)
            cv2.circle(image, (200, y_offset + i * 30 - 10), 8, indicator_color, -1)

# 메인 실행 코드
tracker = MultiColorTracker(['red', 'blue', 'yellow'])
cap = cv2.VideoCapture(0)

print("여러 색상 동시 추적을 시작합니다!")
print("빨간색, 파란색, 노란색 객체를 카메라 앞에 놓아보세요.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 프레임 처리
    result = tracker.process_frame(frame)
    
    cv2.imshow('Multi-Color Tracking', result)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()