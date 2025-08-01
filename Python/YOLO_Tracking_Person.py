import cv2
import numpy as np
from ultralytics import YOLO
from pythonosc import udp_client
import time

class YOLOHumanTracker:
    """
    YOLO를 사용하여 사람을 추적하는 클래스
    단순한 검출을 넘어서 인간의 움직임을 이해하고 해석합니다
    """
    
    def __init__(self, model_size='n'):
        """
        model_size: 'n'(nano), 's'(small), 'm'(medium), 'l'(large), 'x'(extra large)
        더 큰 모델은 더 정확하지만 더 느립니다
        """
        print(f"YOLOv8-pose {model_size} 모델을 로드하고 있습니다...")
        self.model = YOLO(f'yolov8{model_size}-pose.pt')
        
        # OSC 클라이언트 설정
        self.osc_client = udp_client.SimpleUDPClient("127.0.0.1", 5005)
        
        # 추적 관련 변수들
        self.prev_positions = {}  # 이전 프레임의 위치 저장
        self.person_tracks = {}   # 각 사람의 추적 정보
        self.next_person_id = 0   # 새로운 사람에게 할당할 ID
        
        # 관절점 이름 매핑 (COCO 포맷)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        print("YOLO Human Tracker 초기화 완료!")
    
    def process_frame(self, frame):
        """
        프레임을 처리하고 사람을 검출/추적합니다
        이것은 단순한 검출이 아니라, 각 사람의 존재를 인식하고
        그들의 움직임 언어를 해석하는 과정입니다
        """
        # YOLO 추론 실행
        results = self.model(frame, verbose=False)  # verbose=False로 출력 억제
        
        # 시각화를 위한 프레임 복사
        annotated_frame = frame.copy()
        
        # 현재 프레임에서 검출된 사람들
        current_detections = []
        
        if results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy.cpu().numpy()  # 관절점 좌표
            keypoints_confidence = results[0].keypoints.conf.cpu().numpy()  # 신뢰도
            
            # 각 검출된 사람에 대해 처리
            for person_idx, person_keypoints in enumerate(keypoints):
                # 사람의 중심점 계산 (코와 엉덩이의 중점)
                valid_points = []
                for kp_idx, (x, y) in enumerate(person_keypoints):
                    if keypoints_confidence[person_idx][kp_idx] > 0.5:  # 신뢰도 임계값
                        valid_points.append((x, y))
                
                if len(valid_points) > 5:  # 최소 5개 이상의 관절점이 보여야 유효한 검출
                    # 중심점 계산
                    center_x = np.mean([p[0] for p in valid_points])
                    center_y = np.mean([p[1] for p in valid_points])
                    
                    current_detections.append({
                        'center': (center_x, center_y),
                        'keypoints': person_keypoints,
                        'confidence': keypoints_confidence[person_idx],
                        'person_idx': person_idx
                    })
                    
                    # 시각화: 관절점 그리기
                    self._draw_skeleton(annotated_frame, person_keypoints, 
                                      keypoints_confidence[person_idx])
        
        # 사람 추적 (간단한 거리 기반 매칭)
        self._update_tracking(current_detections)
        
        # OSC로 데이터 전송
        self._send_osc_data()
        
        # 추적 정보 시각화
        self._draw_tracking_info(annotated_frame)
        
        return annotated_frame
    
    def _draw_skeleton(self, image, keypoints, confidences):
        """
        사람의 골격을 그립니다
        이것은 단순한 선이 아니라, 인간의 형태를 디지털로 표현하는 것입니다
        """
        # COCO 골격 연결 정의
        skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 머리
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 팔
            (5, 11), (6, 12), (11, 12),  # 몸통
            (11, 13), (13, 15), (12, 14), (14, 16)  # 다리
        ]
        
        # 각 연결선 그리기
        for connection in skeleton_connections:
            kp1_idx, kp2_idx = connection
            
            if (confidences[kp1_idx] > 0.5 and confidences[kp2_idx] > 0.5):
                x1, y1 = int(keypoints[kp1_idx][0]), int(keypoints[kp1_idx][1])
                x2, y2 = int(keypoints[kp2_idx][0]), int(keypoints[kp2_idx][1])
                
                # 신뢰도에 따라 선 색상 결정 (높을수록 초록색)
                confidence = (confidences[kp1_idx] + confidences[kp2_idx]) / 2
                color = (0, int(255 * confidence), int(255 * (1 - confidence)))
                
                cv2.line(image, (x1, y1), (x2, y2), color, 2)
        
        # 관절점 그리기
        for idx, (x, y) in enumerate(keypoints):
            if confidences[idx] > 0.5:
                cv2.circle(image, (int(x), int(y)), 4, (0, 255, 255), -1)
    
    def _update_tracking(self, current_detections):
        """
        검출된 사람들을 추적합니다
        이것은 단순한 ID 할당이 아니라, 각 개인의 연속성을 인식하는 과정입니다
        """
        # 간단한 구현을 위해 가장 가까운 이전 위치와 매칭
        # 실제로는 더 정교한 알고리즘 (예: Deep SORT) 사용 가능
        
        matched = set()
        
        for detection in current_detections:
            center = detection['center']
            min_distance = float('inf')
            matched_id = None
            
            # 기존 추적 중인 사람들과 비교
            for person_id, track_info in self.person_tracks.items():
                if person_id not in matched:
                    prev_center = track_info['center']
                    distance = np.sqrt((center[0] - prev_center[0])**2 + 
                                     (center[1] - prev_center[1])**2)
                    
                    if distance < min_distance and distance < 100:  # 최대 매칭 거리
                        min_distance = distance
                        matched_id = person_id
            
            # 매칭되면 업데이트, 아니면 새 ID 할당
            if matched_id is not None:
                matched.add(matched_id)
                self.person_tracks[matched_id].update({
                    'center': center,
                    'keypoints': detection['keypoints'],
                    'confidence': detection['confidence'],
                    'last_seen': time.time()
                })
            else:
                # 새로운 사람 등록
                new_id = self.next_person_id
                self.next_person_id += 1
                self.person_tracks[new_id] = {
                    'center': center,
                    'keypoints': detection['keypoints'],
                    'confidence': detection['confidence'],
                    'last_seen': time.time(),
                    'created_at': time.time()
                }
        
        # 오래된 추적 제거 (2초 이상 안 보인 경우)
        current_time = time.time()
        to_remove = []
        for person_id, track_info in self.person_tracks.items():
            if current_time - track_info['last_seen'] > 2.0:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.person_tracks[person_id]
            self.osc_client.send_message(f"/tracking/person/{person_id}/lost", [1])
    
    def _send_osc_data(self):
        """
        추적 데이터를 OSC로 전송합니다
        각 사람의 움직임이 디지털 세계에 전달되는 순간입니다
        """
        for person_id, track_info in self.person_tracks.items():
            # 중심점 전송
            center_x = track_info['center'][0] / 640  # 정규화 (카메라 해상도에 맞게 조정)
            center_y = track_info['center'][1] / 480
            self.osc_client.send_message(
                f"/tracking/person/{person_id}/center", 
                [center_x, center_y]
            )
            
            # 각 관절점 전송
            keypoints = track_info['keypoints']
            confidence = track_info['confidence']
            
            for kp_idx, (x, y) in enumerate(keypoints):
                if confidence[kp_idx] > 0.5:
                    norm_x = x / 640
                    norm_y = y / 480
                    kp_name = self.keypoint_names[kp_idx]
                    
                    self.osc_client.send_message(
                        f"/tracking/person/{person_id}/{kp_name}", 
                        [norm_x, norm_y, float(confidence[kp_idx])]
                    )
    
    def _draw_tracking_info(self, image):
        """
        추적 정보를 화면에 표시합니다
        """
        y_offset = 30
        for person_id, track_info in self.person_tracks.items():
            age = time.time() - track_info['created_at']
            text = f"Person {person_id}: {age:.1f}s"
            cv2.putText(image, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25

# 메인 실행 함수
def main():
    # YOLO 추적기 초기화
    tracker = YOLOHumanTracker(model_size='n')  # nano 버전으로 시작
    
    # 웹캠 시작
    cap = cv2.VideoCapture(0)
    
    print("YOLO 기반 인체 추적을 시작합니다!")
    print("ESC: 종료")
    
    # FPS 계산용 변수
    prev_time = time.time()
    fps_list = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # FPS 계산
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        fps_list.append(fps)
        if len(fps_list) > 30:
            fps_list.pop(0)
        avg_fps = sum(fps_list) / len(fps_list)
        
        # 프레임 처리
        annotated_frame = tracker.process_frame(frame)
        
        # FPS 표시
        cv2.putText(annotated_frame, f"FPS: {avg_fps:.1f}", 
                   (frame.shape[1] - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 결과 표시
        cv2.imshow('YOLO Human Tracking', annotated_frame)
        
        if cv2.waitKey(1) == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()