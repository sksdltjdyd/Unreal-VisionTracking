import cv2
import numpy as np
from pythonosc import udp_client
import time
from collections import deque

class MotionAnalyzer:
    """
    객체의 움직임을 분석하는 클래스
    위치, 속도, 가속도를 계산하고 추적합니다
    """
    
    def __init__(self, history_size=10):
        """
        history_size: 속도/가속도 계산을 위해 저장할 과거 위치의 개수
        더 많이 저장하면 더 부드러운 계산이 가능하지만 반응이 느려짐
        """
        self.history_size = history_size
        self.position_history = deque(maxlen=history_size)
        self.time_history = deque(maxlen=history_size)
        self.velocity_history = deque(maxlen=history_size-1)
        
        self.osc_client = udp_client.SimpleUDPClient("127.0.0.1", 5005)
        
        # 현재 상태
        self.current_position = None
        self.current_velocity = None
        self.current_acceleration = None
        self.current_speed = 0
        
    def update(self, x, y, timestamp=None):
        """새로운 위치로 움직임 데이터를 업데이트합니다"""
        if timestamp is None:
            timestamp = time.time()
        
        # 위치 저장
        self.position_history.append((x, y))
        self.time_history.append(timestamp)
        self.current_position = (x, y)
        
        # 최소 2개의 위치가 있어야 속도 계산 가능
        if len(self.position_history) >= 2:
            # 속도 계산 (픽셀/초)
            dt = self.time_history[-1] - self.time_history[-2]
            if dt > 0:  # 0으로 나누기 방지
                dx = self.position_history[-1][0] - self.position_history[-2][0]
                dy = self.position_history[-1][1] - self.position_history[-2][1]
                
                vx = dx / dt
                vy = dy / dt
                
                self.current_velocity = (vx, vy)
                self.velocity_history.append((vx, vy))
                
                # 속력 (스칼라) 계산
                self.current_speed = np.sqrt(vx**2 + vy**2)
                
                # 최소 2개의 속도가 있어야 가속도 계산 가능
                if len(self.velocity_history) >= 2:
                    # 가속도 계산 (픽셀/초²)
                    dvx = self.velocity_history[-1][0] - self.velocity_history[-2][0]
                    dvy = self.velocity_history[-1][1] - self.velocity_history[-2][1]
                    
                    ax = dvx / dt
                    ay = dvy / dt
                    
                    self.current_acceleration = (ax, ay)
    
    def get_motion_state(self):
        """현재 움직임 상태를 분석합니다"""
        if self.current_speed is None:
            return "Unknown"
        
        # 속도에 따른 상태 분류
        if self.current_speed < 50:
            return "Stationary"
        elif self.current_speed < 200:
            return "Slow"
        elif self.current_speed < 500:
            return "Medium"
        else:
            return "Fast"
    
    def get_direction(self):
        """움직임 방향을 계산합니다 (각도)"""
        if self.current_velocity is None:
            return None
        
        vx, vy = self.current_velocity
        if abs(vx) < 0.001 and abs(vy) < 0.001:  # 거의 정지 상태
            return None
        
        # atan2는 -π에서 π 사이의 값을 반환
        # 0도는 오른쪽, 90도는 아래쪽
        angle = np.degrees(np.arctan2(vy, vx))
        return angle

# 색상 검출과 움직임 분석을 결합한 메인 프로그램
def main():
    cap = cv2.VideoCapture(0)
    motion_analyzer = MotionAnalyzer(history_size=15)
    
    # 빨간색 범위
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([1, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 색상 검출 (이전과 동일)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > 500:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # 움직임 분석 업데이트
                    motion_analyzer.update(cx, cy)
                    
                    # OSC로 위치, 속도, 가속도 전송
                    if motion_analyzer.current_position:
                        norm_x = cx / frame.shape[1]
                        norm_y = cy / frame.shape[0]
                        motion_analyzer.osc_client.send_message("/tracking/position", [norm_x, norm_y])
                    
                    if motion_analyzer.current_velocity:
                        vx, vy = motion_analyzer.current_velocity
                        # 속도를 정규화 (화면 크기 기준)
                        norm_vx = vx / frame.shape[1]
                        norm_vy = vy / frame.shape[0]
                        motion_analyzer.osc_client.send_message("/tracking/velocity", [norm_vx, norm_vy])
                        motion_analyzer.osc_client.send_message("/tracking/speed", [motion_analyzer.current_speed])
                    
                    # 시각화
                    cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
                    
                    # 속도 벡터 그리기 (화살표)
                    if motion_analyzer.current_velocity:
                        vx, vy = motion_analyzer.current_velocity
                        # 벡터를 시각화하기 위해 스케일 조정
                        scale = 0.1
                        end_x = int(cx + vx * scale)
                        end_y = int(cy + vy * scale)
                        cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), (0, 255, 0), 2)
                    
                    # 정보 표시
                    y_offset = 30
                    cv2.putText(frame, f"Position: ({cx}, {cy})", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    cv2.putText(frame, f"Speed: {motion_analyzer.current_speed:.1f} px/s", 
                               (10, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    cv2.putText(frame, f"State: {motion_analyzer.get_motion_state()}", 
                               (10, y_offset + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    direction = motion_analyzer.get_direction()
                    if direction is not None:
                        cv2.putText(frame, f"Direction: {direction:.1f}°", 
                                   (10, y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Motion Analysis', frame)
        
        if cv2.waitKey(1) == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()