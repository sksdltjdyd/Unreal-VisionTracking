from ultralytics import YOLO
import cv2

# YOLO 모델 로드 테스트
# 처음 실행 시 모델을 자동으로 다운로드합니다
model = YOLO('yolov8n-pose.pt')  # n은 nano 버전 (가장 가볍고 빠름)

# 웹캠으로 간단한 테스트
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    results = model(frame)
    print(f"검출된 사람 수: {len(results[0].boxes) if results[0].boxes is not None else 0}")
    print("YOLO가 정상적으로 작동합니다!")
cap.release()