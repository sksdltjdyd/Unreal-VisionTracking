// OpenCV OSC Tracking System
// 색상 추적 데이터를 OSC로 언리얼 엔진에 전송

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <deque>
#include <memory>

// OSCPack 라이브러리 포함
#include "C:/Libraries/oscpack/osc/OscOutboundPacketStream.h"
#include "C:/Libraries/oscpack/ip/UdpSocket.h"

// 상수 정의
const int OUTPUT_BUFFER_SIZE = 4096;
const int CAMERA_WIDTH = 1280;
const int CAMERA_HEIGHT = 720;

// ===== OSC 송신자 클래스 =====
class OSCSender {
private:
    UdpTransmitSocket* transmitSocket;
    char buffer[OUTPUT_BUFFER_SIZE];

public:
    // 생성자 - OSC 서버 연결 설정
    OSCSender(const std::string& address = "127.0.0.1", int port = 8000) {
        try {
            // UDP 소켓을 통해 OSC 메시지 전송
            transmitSocket = new UdpTransmitSocket(
                IpEndpointName(address.c_str(), port)
            );
            std::cout << "OSC Sender initialized: " << address << ":" << port << std::endl;
        }
        catch (std::exception& e) {
            std::cerr << "OSC 초기화 실패: " << e.what() << std::endl;
            transmitSocket = nullptr;
        }
    }

    // 소멸자
    ~OSCSender() {
        if (transmitSocket) {
            delete transmitSocket;
        }
    }

    // 모든 트래킹 데이터를 한 번에 전송
    void sendAllTrackingData(
        float mouseX, float mouseY, bool mouseClick,
        float p1X, float p1Y, bool p1Tracking,
        float p2X, float p2Y, bool p2Tracking) {

        if (!transmitSocket) return;

        // OSC 패킷 스트림 생성
        osc::OutboundPacketStream packet(buffer, OUTPUT_BUFFER_SIZE);

        try {
            // OSC 메시지 구성
            // 주소 패턴: /tracking/all
            // 데이터: mouseX, mouseY, mouseClick, p1X, p1Y, p2X, p2Y, p1Track, p2Track
            packet << osc::BeginMessage("/tracking/all")
                << mouseX << mouseY << (mouseClick ? 1 : 0)    // 마우스 데이터
                << p1X << p1Y                                  // Player 1 위치
                << p2X << p2Y                                  // Player 2 위치
                << (p1Tracking ? 1 : 0)                        // Player 1 추적 상태
                << (p2Tracking ? 1 : 0)                        // Player 2 추적 상태
                << osc::EndMessage;

            // 패킷 전송
            transmitSocket->Send(packet.Data(), packet.Size());

        }
        catch (std::exception& e) {
            std::cerr << "OSC 전송 오류: " << e.what() << std::endl;
        }
    }

    // 개별 데이터 전송 메서드들 (선택적 사용)
    void sendMouseData(float x, float y, bool click) {
        if (!transmitSocket) return;

        osc::OutboundPacketStream packet(buffer, OUTPUT_BUFFER_SIZE);
        packet << osc::BeginMessage("/tracking/mouse")
            << x << y << (click ? 1 : 0)
            << osc::EndMessage;

        transmitSocket->Send(packet.Data(), packet.Size());
    }

    void sendPlayerData(int playerNum, float x, float y, bool tracking) {
        if (!transmitSocket) return;

        osc::OutboundPacketStream packet(buffer, OUTPUT_BUFFER_SIZE);
        std::string address = (playerNum == 1) ? "/tracking/player1" : "/tracking/player2";

        packet << osc::BeginMessage(address.c_str())
            << x << y << (tracking ? 1 : 0)
            << osc::EndMessage;

        transmitSocket->Send(packet.Data(), packet.Size());
    }

    // 제스처 전송
    void sendGesture(const std::string& gesture) {
        if (!transmitSocket) return;

        osc::OutboundPacketStream packet(buffer, OUTPUT_BUFFER_SIZE);
        packet << osc::BeginMessage("/tracking/gesture")
            << gesture.c_str()
            << osc::EndMessage;

        transmitSocket->Send(packet.Data(), packet.Size());
    }
};

// ===== 간단한 칼만 필터 (부드러운 추적용) =====
class SimpleKalmanFilter {
private:
    cv::Point2f position;
    cv::Point2f velocity;
    bool initialized;
    float alpha;  // 스무딩 팩터

public:
    SimpleKalmanFilter(float smoothing = 0.7f)
        : position(0, 0), velocity(0, 0), initialized(false), alpha(smoothing) {
    }

    cv::Point2f update(const cv::Point2f& measurement, bool detected) {
        if (!initialized && detected) {
            position = measurement;
            initialized = true;
            return position;
        }

        if (detected) {
            // 새로운 측정값이 있을 때
            cv::Point2f newPos = measurement;
            velocity = newPos - position;

            // 스무딩 적용 (Low-pass filter)
            position = position * alpha + newPos * (1.0f - alpha);
        }
        else {
            // 측정값이 없을 때는 예측
            position += velocity * 0.9f;  // 속도 감쇠 적용
        }

        return position;
    }

    bool isTracking() const { return initialized; }
    void reset() { initialized = false; velocity = cv::Point2f(0, 0); }
};

// ===== 메인 트래킹 시스템 =====
class OSCTrackingSystem {
private:
    // OSC 송신자
    OSCSender oscSender;

    // 칼만 필터들 (각 추적 대상별)
    SimpleKalmanFilter mouseFilter;
    SimpleKalmanFilter player1Filter;
    SimpleKalmanFilter player2Filter;

    // 제스처 인식용
    std::deque<cv::Point2f> gesturePoints;

    // HSV 색상 범위 설정
    struct ColorRange {
        cv::Scalar lower;
        cv::Scalar upper;
        std::string name;
    };

    ColorRange mouseColor;    // 초록색
    ColorRange player1Color;  // 빨간색
    ColorRange player2Color;  // 노란색

    // FPS 계산용
    std::chrono::steady_clock::time_point lastTime;
    float fps;

public:
    // 생성자 - 기본 색상 설정
    OSCTrackingSystem() : oscSender("127.0.0.1", 8000), fps(0) {
        // 초록색 마우스
        mouseColor.lower = cv::Scalar(40, 100, 100);
        mouseColor.upper = cv::Scalar(80, 255, 255);
        mouseColor.name = "Mouse (Green)";

        // 빨간색 Player 1
        player1Color.lower = cv::Scalar(0, 120, 100);
        player1Color.upper = cv::Scalar(10, 255, 255);
        player1Color.name = "Player 1 (Red)";

        // 노란색 Player 2
        player2Color.lower = cv::Scalar(20, 120, 100);
        player2Color.upper = cv::Scalar(40, 255, 255);
        player2Color.name = "Player 2 (Yellow)";

        lastTime = std::chrono::steady_clock::now();
    }

    // 색상 감지 함수
    cv::Point2f detectColor(const cv::Mat& hsv, const ColorRange& color,
        cv::Mat& outputMask, int& area) {
        // 색상 범위로 마스크 생성
        cv::inRange(hsv, color.lower, color.upper, outputMask);

        // 노이즈 제거 (모폴로지 연산)
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(outputMask, outputMask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(outputMask, outputMask, cv::MORPH_CLOSE, kernel);

        // 컨투어 찾기
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(outputMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 가장 큰 컨투어 찾기
        double maxArea = 0;
        int maxIdx = -1;

        for (size_t i = 0; i < contours.size(); i++) {
            double currentArea = cv::contourArea(contours[i]);
            if (currentArea > maxArea && currentArea > 500) {  // 최소 크기 필터
                maxArea = currentArea;
                maxIdx = static_cast<int>(i);
            }
        }

        // 중심점 계산
        if (maxIdx >= 0) {
            area = static_cast<int>(maxArea);
            cv::Moments m = cv::moments(contours[maxIdx]);
            if (m.m00 != 0) {
                return cv::Point2f(
                    static_cast<float>(m.m10 / m.m00),
                    static_cast<float>(m.m01 / m.m00)
                );
            }
        }

        area = 0;
        return cv::Point2f(-1, -1);  // 감지 실패
    }

    // 간단한 제스처 인식
    std::string detectGesture() {
        if (gesturePoints.size() < 20) return "NONE";

        // 전체 이동 거리 계산
        float totalDistance = 0;
        float totalDx = 0, totalDy = 0;

        for (size_t i = 1; i < gesturePoints.size(); i++) {
            cv::Point2f diff = gesturePoints[i] - gesturePoints[i - 1];
            totalDistance += cv::norm(diff);
            totalDx += std::abs(diff.x);
            totalDy += std::abs(diff.y);
        }

        // 제스처 판별
        if (totalDistance < 50) return "NONE";  // 움직임이 너무 작음

        // 수직/수평 움직임 판별
        if (totalDy > totalDx * 2 && totalDy > 100) return "VERTICAL";
        if (totalDx > totalDy * 2 && totalDx > 100) return "HORIZONTAL";

        // 원형 움직임 판별
        cv::Point2f startEnd = gesturePoints.back() - gesturePoints.front();
        float endDistance = cv::norm(startEnd);
        if (endDistance < 50 && totalDistance > 200) return "CIRCLE";

        return "NONE";
    }

    // 메인 프레임 처리 함수
    void processFrame(const cv::Mat& frame) {
        // BGR을 HSV로 변환 (색상 감지가 더 정확함)
        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        // 1. 마우스(초록색) 추적
        cv::Mat mouseMask;
        int mouseArea;
        cv::Point2f mouseRaw = detectColor(hsv, mouseColor, mouseMask, mouseArea);
        cv::Point2f mousePos = mouseFilter.update(mouseRaw, mouseRaw.x > 0);

        // 클릭 감지 (주먹을 쥐면 면적이 줄어듦)
        bool mouseClick = (mouseArea > 0 && mouseArea < 2000);

        // 2. Player 1 (빨간색) 추적
        cv::Mat p1Mask;
        int p1Area;
        cv::Point2f p1Raw = detectColor(hsv, player1Color, p1Mask, p1Area);
        cv::Point2f p1Pos = player1Filter.update(p1Raw, p1Raw.x > 0);

        // 3. Player 2 (노란색) 추적
        cv::Mat p2Mask;
        int p2Area;
        cv::Point2f p2Raw = detectColor(hsv, player2Color, p2Mask, p2Area);
        cv::Point2f p2Pos = player2Filter.update(p2Raw, p2Raw.x > 0);

        // 4. 제스처 추적 (마우스 궤적)
        if (mouseFilter.isTracking()) {
            gesturePoints.push_back(mousePos);
            if (gesturePoints.size() > 30) {
                gesturePoints.pop_front();
            }
        }

        // 5. 좌표 정규화 (0~1 범위로)
        float normMouseX = mousePos.x / CAMERA_WIDTH;
        float normMouseY = mousePos.y / CAMERA_HEIGHT;
        float normP1X = p1Pos.x / CAMERA_WIDTH;
        float normP1Y = p1Pos.y / CAMERA_HEIGHT;
        float normP2X = p2Pos.x / CAMERA_WIDTH;
        float normP2Y = p2Pos.y / CAMERA_HEIGHT;

        // 6. OSC로 데이터 전송
        oscSender.sendAllTrackingData(
            normMouseX, normMouseY, mouseClick,
            normP1X, normP1Y, player1Filter.isTracking(),
            normP2X, normP2Y, player2Filter.isTracking()
        );

        // 제스처 감지 및 전송
        std::string gesture = detectGesture();
        if (gesture != "NONE") {
            oscSender.sendGesture(gesture);
            gesturePoints.clear();  // 제스처 인식 후 초기화
        }

        // 7. 디버그 시각화
        drawDebugInfo(frame, mousePos, p1Pos, p2Pos,
            mouseClick, mouseFilter.isTracking(),
            player1Filter.isTracking(), player2Filter.isTracking());
    }

    // 디버그 정보 그리기
    void drawDebugInfo(cv::Mat& frame,
        const cv::Point2f& mousePos,
        const cv::Point2f& p1Pos,
        const cv::Point2f& p2Pos,
        bool mouseClick, bool mouseTracking,
        bool p1Tracking, bool p2Tracking) {

        // FPS 계산
        auto now = std::chrono::steady_clock::now();
        float deltaTime = std::chrono::duration<float>(now - lastTime).count();
        fps = 0.9f * fps + 0.1f * (1.0f / deltaTime);
        lastTime = now;

        // 마우스 그리기
        if (mouseTracking) {
            cv::circle(frame, cv::Point(mousePos), 20,
                mouseClick ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 200, 0), 3);
            cv::putText(frame, mouseClick ? "CLICK!" : "Mouse",
                cv::Point(mousePos) + cv::Point(-20, -25),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

            // 제스처 궤적 그리기
            for (size_t i = 1; i < gesturePoints.size(); i++) {
                cv::line(frame, gesturePoints[i - 1], gesturePoints[i],
                    cv::Scalar(0, 255, 0), 2);
            }
        }

        // Player 1 그리기
        if (p1Tracking) {
            cv::circle(frame, cv::Point(p1Pos), 25, cv::Scalar(0, 0, 255), 3);
            cv::putText(frame, "P1", cv::Point(p1Pos) + cv::Point(-10, 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        }

        // Player 2 그리기
        if (p2Tracking) {
            cv::circle(frame, cv::Point(p2Pos), 25, cv::Scalar(0, 255, 255), 3);
            cv::putText(frame, "P2", cv::Point(p2Pos) + cv::Point(-10, 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
        }

        // 상태 정보 표시
        cv::putText(frame, "OSC -> Unreal Engine 5.4", cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

        std::stringstream status;
        status << "FPS: " << static_cast<int>(fps)
            << " | Port: 8000"
            << " | Mouse: " << (mouseTracking ? "OK" : "NO")
            << " | P1: " << (p1Tracking ? "OK" : "NO")
            << " | P2: " << (p2Tracking ? "OK" : "NO");

        cv::putText(frame, status.str(), cv::Point(10, 60),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);
    }

    // HSV 조절용 트랙바 생성
    void createTrackbars() {
        cv::namedWindow("HSV Controls", cv::WINDOW_NORMAL);

        // 마우스 (초록색) 조절
        cv::createTrackbar("Mouse H Min", "HSV Controls", &mouseColor.lower[0], 180);
        cv::createTrackbar("Mouse H Max", "HSV Controls", &mouseColor.upper[0], 180);
        cv::createTrackbar("Mouse S Min", "HSV Controls", &mouseColor.lower[1], 255);
        cv::createTrackbar("Mouse S Max", "HSV Controls", &mouseColor.upper[1], 255);
        cv::createTrackbar("Mouse V Min", "HSV Controls", &mouseColor.lower[2], 255);
        cv::createTrackbar("Mouse V Max", "HSV Controls", &mouseColor.upper[2], 255);

        // Player 1 (빨간색) 조절
        cv::createTrackbar("P1 H Min", "HSV Controls", &player1Color.lower[0], 180);
        cv::createTrackbar("P1 H Max", "HSV Controls", &player1Color.upper[0], 180);
        cv::createTrackbar("P1 S Min", "HSV Controls", &player1Color.lower[1], 255);
        cv::createTrackbar("P1 S Max", "HSV Controls", &player1Color.upper[1], 255);

        // Player 2 (노란색) 조절
        cv::createTrackbar("P2 H Min", "HSV Controls", &player2Color.lower[0], 180);
        cv::createTrackbar("P2 H Max", "HSV Controls", &player2Color.upper[0], 180);
        cv::createTrackbar("P2 S Min", "HSV Controls", &player2Color.lower[1], 255);
        cv::createTrackbar("P2 S Max", "HSV Controls", &player2Color.upper[1], 255);
    }
};

// ===== 메인 함수 =====
int main() {
    std::cout << "=== OpenCV OSC Tracking System ===" << std::endl;
    std::cout << "Sending to: osc://localhost:8000" << std::endl;
    std::cout << "Green: Mouse (make fist to click)" << std::endl;
    std::cout << "Red: Player 1" << std::endl;
    std::cout << "Yellow: Player 2" << std::endl;
    std::cout << "Press ESC to exit" << std::endl << std::endl;

    // 카메라 열기
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open camera!" << std::endl;
        return -1;
    }

    // 카메라 설정
    cap.set(cv::CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT);
    cap.set(cv::CAP_PROP_FPS, 60);

    // 트래킹 시스템 생성
    OSCTrackingSystem tracker;
    tracker.createTrackbars();

    // 메인 윈도우
    cv::namedWindow("OSC Tracking", cv::WINDOW_NORMAL);

    // 메인 루프
    cv::Mat frame;
    while (true) {
        // 프레임 캡처
        cap >> frame;
        if (frame.empty()) continue;

        // 좌우 반전 (거울 효과)
        cv::flip(frame, frame, 1);

        // 프레임 처리
        tracker.processFrame(frame);

        // 화면 표시
        cv::imshow("OSC Tracking", frame);

        // 키 입력 처리
        char key = cv::waitKey(1);
        if (key == 27) break;  // ESC

        // 'd' 키로 디버그 정보 출력
        if (key == 'd' || key == 'D') {
            std::cout << "Debug: Check OSC messages in Unreal Engine" << std::endl;
        }
    }

    // 정리
    cap.release();
    cv::destroyAllWindows();

    std::cout << "Program ended." << std::endl;
    return 0;
}