import cv2  # OpenCV 라이브러리

# 웹캠 연결하기
# 0은 기본 웹캠을 의미 웹캠이 여러 개면 1, 2 등을 사용
cap = cv2.VideoCapture(0)

# 웹캠이 제대로 열렸는지 확인
if not cap.isOpened():
    print("웹캠을 열 수 없습니다")
    exit()

print("웹캠이 성공적으로 열렸습니다. ESC 키를 누르면 종료됩니다.")

# 계속해서 영상을 읽고 표시하기
while True:
    # 한 프레임 읽기
    # ret: 성공적으로 읽었는지 여부 (True/False)
    # frame: 실제 이미지 데이터
    ret, frame = cap.read()
    
    # 프레임을 제대로 읽지 못했다면 종료
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break
    
    # 화면에 영상 표시
    cv2.imshow('Webcam Feed', frame)
    
    # 1밀리초 동안 키 입력 대기
    # ESC 키(27번)를 누르면 종료
    if cv2.waitKey(1) == 27:
        break

# 자원 정리
cap.release()
cv2.destroyAllWindows()