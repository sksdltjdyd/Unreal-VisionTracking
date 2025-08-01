import cv2
import numpy as np
from pythonosc import udp_client
import time
from collections import deque
import math

class ColorMotionTracker:
    """
    특정 색상 하나의 움직임을 추적하는 클래스
    위치, 속도, 가속도를 개별적으로 관리합니다
    """
    
    def __init__(self, color_name, color_ranges, history_size=15):
        self.color_name = color_name
        self.color_ranges = color_ranges
        self.history_size = history_size
        
        # 위치 및 시간 기록
        self.position_history = deque(maxlen=history_size)
        self.time_history = deque(maxlen=history_size)
        
        # 현재 상태 정보
        self.current_position = None
        self.current_velocity = (0, 0)
        self.current_acceleration = (0, 0)
        self.current_speed = 0
        self.is_detected = False
        self.last_detected_time = 0
        
        # 부드러운 움직임을 위한 필터링
        self.smoothed_position = None
        self.smoothing_factor = 0.3
        
        # 궤적 표시를 위한 점들
        self.trail_points = deque(maxlen=50)
        
    def update(self, position=None, timestamp=None):
        """
        새로운 위치로 상태를 업데이트합니다
        position이 None이면 객체가 감지되지 않은 것으로 처리
        """
        if timestamp is None:
            timestamp = time.time()
        
        if position is not None:
            x, y = position
            self.is_detected = True
            self.last_detected_time = timestamp
            
            # 첫 번째 감지이거나 오랜만에 다시 감지된 경우
            if self.smoothed_position is None or (timestamp - self.last_detected_time) > 1.0:
                self.smoothed_position = (x, y)
            else:
                # 부드러운 위치 계산 (노이즈 감소)
                smooth_x = self.smoothed_position[0] + (x - self.smoothed_position[0]) * self.smoothing_factor
                smooth_y = self.smoothed_position[1] + (y - self.smoothed_position[1]) * self.smoothing_factor
                self.smoothed_position = (smooth_x, smooth_y)
            
            # 기록 저장
            self.position_history.append(self.smoothed_position)
            self.time_history.append(timestamp)
            self.current_position = self.smoothed_position
            self.trail_points.append(self.smoothed_position)
            
            # 속도 계산
            self._calculate_velocity()
            
            # 가속도 계산
            self._calculate_acceleration()
            
        else:
            self.is_detected = False
            # 일정 시간 이상 감지되지 않으면 속도를 0으로 감소
            if timestamp - self.last_detected_time > 0.5:
                decay_factor = 0.9
                self.current_velocity = (
                    self.current_velocity[0] * decay_factor,
                    self.current_velocity[1] * decay_factor
                )
                self.current_speed *= decay_factor
    
    def _calculate_velocity(self):
        """속도를 계산합니다"""
        if len(self.position_history) < 2:
            return
        
        # 최근 몇 개의 점을 사용해서 평균 속도 계산 (노이즈 감소)
        num_points = min(3, len(self.position_history))
        
        total_vx, total_vy = 0, 0
        valid_count = 0
        
        for i in range(1, num_points):
            dt = self.time_history[-i] - self.time_history[-i-1]
            if dt > 0:
                dx = self.position_history[-i][0] - self.position_history[-i-1][0]
                dy = self.position_history[-i][1] - self.position_history[-i-1][1]
                
                vx = dx / dt
                vy = dy / dt
                
                total_vx += vx
                total_vy += vy
                valid_count += 1
        
        if valid_count > 0:
            self.current_velocity = (total_vx / valid_count, total_vy / valid_count)
            self.current_speed = math.sqrt(self.current_velocity[0]**2 + self.current_velocity[1]**2)
    
    def _calculate_acceleration(self):
        """가속도를 계산합니다"""
        if len(self.position_history) < 3:
            return
        
        # 이전 속도와 현재 속도의 차이로 가속도 계산
        dt = self.time_history[-1] - self.time_history[-2]
        if dt > 0:
            # 간단한 차분으로 가속도 근사
            prev_vx = (self.position_history[-2][0] - self.position_history[-3][0]) / (self.time_history[-2] - self.time_history[-3])
            prev_vy = (self.position_history[-2][1] - self.position_history[-3][1]) / (self.time_history[-2] - self.time_history[-3])
            
            ax = (self.current_velocity[0] - prev_vx) / dt
            ay = (self.current_velocity[1] - prev_vy) / dt
            
            self.current_acceleration = (ax, ay)
    
    def get_direction_angle(self):
        """움직임 방향을 각도로 반환합니다 (0-360도)"""
        if self.current_speed < 10:  # 거의 정지 상태
            return None
        
        vx, vy = self.current_velocity
        angle = math.degrees(math.atan2(vy, vx))
        # -180~180을 0~360으로 변환
        return (angle + 360) % 360
    
    def get_motion_state(self):
        """현재 움직임 상태를 문자열로 반환합니다"""
        if not self.is_detected:
            return "Lost"
        elif self.current_speed < 50:
            return "Stationary"
        elif self.current_speed < 200:
            return "Slow"
        elif self.current_speed < 500:
            return "Medium"
        else:
            return "Fast"


class MultiColorMotionTracker:
    """
    여러 색상을 동시에 추적하며 각각의 움직임을 분석하는 통합 클래스
    """
    
    def __init__(self, colors_config):
        """
        colors_config: 색상 이름과 범위를 포함한 딕셔너리
        """
        self.colors_config = colors_config
        self.trackers = {}
        self.osc_client = udp_client.SimpleUDPClient("127.0.0.1", 5005)
        
        # 각 색상별로 독립적인 추적기 생성
        for color_name, color_info in colors_config.items():
            self.trackers[color_name] = ColorMotionTracker(color_name, color_info)
        
        # 상호작용 분석을 위한 변수
        self.interaction_threshold = 150  # 픽셀 단위
        
    def process_frame(self, frame):
        """
        프레임을 처리하고 모든 색상을 추적합니다
        """
        # 전처리 (한 번만 수행)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # 결과 표시용 이미지
        display = frame.copy()
        
        # 각 색상별로 처리
        current_time = time.time()
        detected_colors = []
        
        for color_name, tracker in self.trackers.items():
            # 마스크 생성
            mask = self._create_color_mask(hsv, color_name)
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            position = None
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > 300:
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        position = (cx, cy)
                        detected_colors.append(color_name)
            
            # 추적기 업데이트
            tracker.update(position, current_time)
            
            # OSC 메시지 전송
            self._send_osc_data(color_name, tracker)
            
            # 시각화
            self._visualize_tracker(display, tracker)
        
        # 상호작용 분석
        if len(detected_colors) >= 2:
            self._analyze_interactions(display)
        
        # 전체 상태 정보 표시
        self._draw_status_panel(display)
        
        return display
    
    def _create_color_mask(self, hsv_image, color_name):
        """특정 색상에 대한 마스크를 생성합니다"""
        color_info = self.colors_config[color_name]
        mask = None
        
        for i in range(len(color_info['lower'])):
            lower = np.array(color_info['lower'][i])
            upper = np.array(color_info['upper'][i])
            current_mask = cv2.inRange(hsv_image, lower, upper)
            
            if mask is None:
                mask = current_mask
            else:
                mask = cv2.bitwise_or(mask, current_mask)
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _send_osc_data(self, color_name, tracker):
        """추적 데이터를 OSC로 전송합니다"""
        if tracker.is_detected and tracker.current_position:
            # 위치 (정규화된 좌표)
            norm_x = tracker.current_position[0] / 640  # 카메라 해상도에 맞게 조정
            norm_y = tracker.current_position[1] / 480
            self.osc_client.send_message(f"/tracking/{color_name}/position", [norm_x, norm_y])
            
            # 속도
            norm_vx = tracker.current_velocity[0] / 640
            norm_vy = tracker.current_velocity[1] / 480
            self.osc_client.send_message(f"/tracking/{color_name}/velocity", [norm_vx, norm_vy])
            
            # 속력
            self.osc_client.send_message(f"/tracking/{color_name}/speed", [tracker.current_speed])
            
            # 방향
            direction = tracker.get_direction_angle()
            if direction is not None:
                self.osc_client.send_message(f"/tracking/{color_name}/direction", [direction])
            
            # 상태
            state = tracker.get_motion_state()
            self.osc_client.send_message(f"/tracking/{color_name}/state", [state])
    
    def _visualize_tracker(self, image, tracker):
        """각 추적기의 상태를 시각화합니다"""
        if not tracker.is_detected:
            return
        
        color_bgr = self.colors_config[tracker.color_name]['rgb']
        
        # 궤적 그리기
        if len(tracker.trail_points) > 1:
            for i in range(1, len(tracker.trail_points)):
                # 시간에 따라 투명도 변화
                alpha = i / len(tracker.trail_points)
                thickness = int(1 + alpha * 3)
                
                # 궤적 색상은 원래 색상의 어두운 버전
                trail_color = tuple(int(c * 0.6) for c in color_bgr)
                
                pt1 = tuple(map(int, tracker.trail_points[i-1]))
                pt2 = tuple(map(int, tracker.trail_points[i]))
                cv2.line(image, pt1, pt2, trail_color, thickness)
        
        # 현재 위치
        if tracker.current_position:
            x, y = map(int, tracker.current_position)
            
            # 중심점
            cv2.circle(image, (x, y), 8, color_bgr, -1)
            cv2.circle(image, (x, y), 12, color_bgr, 2)
            
            # 속도 벡터 그리기
            if tracker.current_speed > 50:  # 움직이고 있을 때만
                vx, vy = tracker.current_velocity
                scale = 0.1
                end_x = int(x + vx * scale)
                end_y = int(y + vy * scale)
                cv2.arrowedLine(image, (x, y), (end_x, end_y), (0, 255, 0), 2)
            
            # 정보 텍스트
            info = f"{tracker.color_name}: {tracker.current_speed:.0f} px/s"
            cv2.putText(image, info, (x + 15, y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)
    
    def _analyze_interactions(self, image):
        """
        여러 객체 간의 상호작용을 분석합니다
        예: 거리, 상대 속도, 충돌 예측 등
        """
        active_trackers = [(name, t) for name, t in self.trackers.items() if t.is_detected]
        
        # 모든 쌍에 대해 분석
        for i in range(len(active_trackers)):
            for j in range(i + 1, len(active_trackers)):
                name1, tracker1 = active_trackers[i]
                name2, tracker2 = active_trackers[j]
                
                if tracker1.current_position and tracker2.current_position:
                    # 거리 계산
                    x1, y1 = tracker1.current_position
                    x2, y2 = tracker2.current_position
                    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    
                    # 가까운 경우 연결선 그리기
                    if distance < self.interaction_threshold:
                        # 거리에 따라 선 색상 변경 (가까울수록 빨간색)
                        intensity = 1 - (distance / self.interaction_threshold)
                        line_color = (0, int(255 * (1 - intensity)), int(255 * intensity))
                        
                        cv2.line(image, 
                                tuple(map(int, tracker1.current_position)), 
                                tuple(map(int, tracker2.current_position)), 
                                line_color, 2)
                        
                        # 중간점에 거리 표시
                        mid_x = int((x1 + x2) / 2)
                        mid_y = int((y1 + y2) / 2)
                        cv2.putText(image, f"{distance:.0f}px", 
                                   (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.5, (255, 255, 255), 1)
                        
                        # 상호작용 이벤트 OSC 전송
                        self.osc_client.send_message(
                            f"/interaction/{name1}_{name2}/distance", 
                            [distance / self.interaction_threshold]
                        )
    
    def _draw_status_panel(self, image):
        """상태 패널을 그립니다"""
        # 반투명 배경
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (250, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # 각 색상의 상태 표시
        y_offset = 30
        for i, (color_name, tracker) in enumerate(self.trackers.items()):
            color_bgr = self.colors_config[color_name]['rgb']
            state = tracker.get_motion_state()
            
            # 상태 인디케이터
            indicator_color = color_bgr if tracker.is_detected else (100, 100, 100)
            cv2.circle(image, (25, y_offset + i * 25), 6, indicator_color, -1)
            
            # 텍스트 정보
            text = f"{color_name}: {state}"
            if tracker.is_detected:
                text += f" ({tracker.current_speed:.0f} px/s)"
            
            cv2.putText(image, text, (40, y_offset + i * 25 + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# 메인 실행 함수
def main():
    # 색상 설정 (실제 환경에 맞게 조정 필요)
    colors_config = {
        'red': {
            'lower': [(0, 120, 70), (170, 120, 70)],
            'upper': [(1, 255, 255), (180, 255, 255)],
            'rgb': (0, 0, 255)
        },
        'blue': {
            'lower': [(100, 120, 70)],
            'upper': [(130, 255, 255)],
            'rgb': (255, 0, 0)
        },
        'yellow': {
        'lower': [(20, 100, 100)],
        'upper': [(30, 255, 255)],
        'rgb': (0, 255, 255)
        }
    }
    
    # 멀티 컬러 모션 트래커 생성
    tracker = MultiColorMotionTracker(colors_config)
    
    # 웹캠 시작
    cap = cv2.VideoCapture(0)
    
    print("멀티 컬러 속도 추적 시스템")
    print("빨간색, 파란색, 노란색 객체를 동시에 추적합니다.")
    print("ESC: 종료")
    
    # FPS 계산용
    prev_time = time.time()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # FPS 계산
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # 프레임 처리
        result = tracker.process_frame(frame)
        
        # FPS 표시
        cv2.putText(result, f"FPS: {fps:.1f}", 
                   (frame.shape[1] - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Multi-Color Motion Tracking', result)
        
        if cv2.waitKey(1) == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()