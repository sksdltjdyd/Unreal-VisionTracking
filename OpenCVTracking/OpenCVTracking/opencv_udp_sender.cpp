// opencv_to_unreal_sender.cpp
// OpenCV 트래킹 데이터를 언리얼 엔진 5.4로 UDP 전송

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <chrono>
#include <thread>
#include <iomanip>

#ifdef _WIN32
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

// JSON 형식으로 데이터 구조화
struct TrackingData {
    // 타임스탬프
    float timestamp;

    // 마우스 데이터
    float mouseX;
    float mouseY;
    bool mouseClick;
    bool mouseTracking;

    // Player 1 데이터
    float player1X;
    float player1Y;
    bool player1Tracking;

    // Player 2 데이터
    float player2X;
    float player2Y;
    bool player2Tracking;

    // 게임 데이터
    int player1Score;
    int player2Score;
    std::string gameState;
    std::string gesture;

    // JSON 형식으로 변환
    std::string toJSON() const {
        std::stringstream ss;
        ss << "{";
        ss << "\"timestamp\":" << std::fixed << std::setprecision(3) << timestamp << ",";
        ss << "\"mouse\":{";
        ss << "\"x\":" << mouseX << ",";
        ss << "\"y\":" << mouseY << ",";
        ss << "\"click\":" << (mouseClick ? "true" : "false") << ",";
        ss << "\"tracking\":" << (mouseTracking ? "true" : "false");
        ss << "},";
        ss << "\"player1\":{";
        ss << "\"x\":" << player1X << ",";
        ss << "\"y\":" << player1Y << ",";
        ss << "\"tracking\":" << (player1Tracking ? "true" : "false");
        ss << "},";
        ss << "\"player2\":{";
        ss << "\"x\":" << player2X << ",";
        ss << "\"y\":" << player2Y << ",";
        ss << "\"tracking\":" << (player2Tracking ? "true" : "false");
        ss << "},";
        ss << "\"game\":{";
        ss << "\"player1Score\":" << player1Score << ",";
        ss << "\"player2Score\":" << player2Score << ",";
        ss << "\"state\":\"" << gameState << "\",";
        ss << "\"gesture\":\"" << gesture << "\"";
        ss << "}}";
        return ss.str();
    }
};

class UDPSender {
private:
    SOCKET sock;
    struct sockaddr_in server_addr;
    bool initialized;

public:
    UDPSender() : sock(INVALID_SOCKET), initialized(false) {}

    bool init(const std::string& ip, int port) {
#ifdef _WIN32
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            std::cerr << "WSAStartup failed" << std::endl;
            return false;
        }
#endif

        sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (sock == INVALID_SOCKET) {
            std::cerr << "Socket creation failed" << std::endl;
            return false;
        }

        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(port);

        if (inet_pton(AF_INET, ip.c_str(), &server_addr.sin_addr) != 1) {
            std::cerr << "Invalid address" << std::endl;
            closesocket(sock);
            return false;
        }

        initialized = true;
        std::cout << "UDP Sender initialized: " << ip << ":" << port << std::endl;
        return true;
    }

    void sendData(const TrackingData& data) {
        if (!initialized) return;

        std::string jsonData = data.toJSON();

        int result = sendto(sock, jsonData.c_str(), static_cast<int>(jsonData.length()), 0,
            reinterpret_cast<struct sockaddr*>(&server_addr), sizeof(server_addr));

        if (result == SOCKET_ERROR) {
#ifdef _WIN32
            std::cerr << "Send failed: " << WSAGetLastError() << std::endl;
#endif
        }
    }

    ~UDPSender() {
        if (sock != INVALID_SOCKET) {
#ifdef _WIN32
            closesocket(sock);
            WSACleanup();
#else
            close(sock);
#endif
        }
    }
};

// 간단한 칼만 필터 (기존 코드에서)
class SimpleKalmanFilter {
public:
    cv::Point2f position;
    cv::Point2f velocity;
    bool initialized;

    SimpleKalmanFilter() : position(0, 0), velocity(0, 0), initialized(false) {}

    cv::Point2f update(const cv::Point2f& measurement, bool found) {
        if (!initialized && found) {
            position = measurement;
            initialized = true;
            return position;
        }

        if (found) {
            // 간단한 low-pass filter
            cv::Point2f newPos = measurement;
            velocity = newPos - position;
            position = position * 0.7f + newPos * 0.3f;
        }
        else {
            // 예측
            position += velocity * 0.9f;
        }

        return position;
    }
};

// 메인 트래킹 시스템
class OpenCVTracker {
private:
    UDPSender udpSender;
    TrackingData currentData;

    SimpleKalmanFilter mouseFilter;
    SimpleKalmanFilter player1Filter;
    SimpleKalmanFilter player2Filter;

    std::chrono::steady_clock::time_point startTime;

public:
    // HSV 범위
    cv::Scalar mouseColorLower = cv::Scalar(40, 100, 100);   // 초록
    cv::Scalar mouseColorUpper = cv::Scalar(80, 255, 255);
    cv::Scalar player1ColorLower = cv::Scalar(0, 120, 100);  // 빨강
    cv::Scalar player1ColorUpper = cv::Scalar(10, 255, 255);
    cv::Scalar player2ColorLower = cv::Scalar(20, 120, 100); // 노랑
    cv::Scalar player2ColorUpper = cv::Scalar(40, 255, 255);

    OpenCVTracker() {
        udpSender.init("127.0.0.1", 7777);
        startTime = std::chrono::steady_clock::now();
        currentData.gameState = "playing";
        currentData.player1Score = 0;
        currentData.player2Score = 0;
    }

    cv::Point2f detectColor(const cv::Mat& hsv, const cv::Scalar& lower,
        const cv::Scalar& upper, int& area) {
        cv::Mat mask;
        cv::inRange(hsv, lower, upper, mask);

        cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), 2);
        cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), 2);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if (!contours.empty()) {
            double maxArea = 0;
            int maxIdx = -1;

            for (size_t i = 0; i < contours.size(); i++) {
                double a = cv::contourArea(contours[i]);
                if (a > maxArea && a > 500) {
                    maxArea = a;
                    maxIdx = i;
                }
            }

            if (maxIdx >= 0) {
                area = static_cast<int>(maxArea);
                cv::Moments m = cv::moments(contours[maxIdx]);
                return cv::Point2f(m.m10 / m.m00, m.m01 / m.m00);
            }
        }

        area = 0;
        return cv::Point2f(-1, -1);
    }

    void processFrame(const cv::Mat& frame) {
        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        // 타임스탬프
        auto now = std::chrono::steady_clock::now();
        currentData.timestamp = std::chrono::duration<float>(now - startTime).count();

        // 마우스 감지
        int mouseArea;
        cv::Point2f mouseRaw = detectColor(hsv, mouseColorLower, mouseColorUpper, mouseArea);
        cv::Point2f mousePos = mouseFilter.update(mouseRaw, mouseRaw.x > 0);

        currentData.mouseX = mousePos.x / frame.cols;  // 0-1로 정규화
        currentData.mouseY = mousePos.y / frame.rows;
        currentData.mouseClick = (mouseArea > 0 && mouseArea < 2000);
        currentData.mouseTracking = mouseFilter.initialized;

        // Player 1 감지
        int p1Area;
        cv::Point2f p1Raw = detectColor(hsv, player1ColorLower, player1ColorUpper, p1Area);
        cv::Point2f p1Pos = player1Filter.update(p1Raw, p1Raw.x > 0);

        currentData.player1X = p1Pos.x / frame.cols;
        currentData.player1Y = p1Pos.y / frame.rows;
        currentData.player1Tracking = player1Filter.initialized;

        // Player 2 감지
        int p2Area;
        cv::Point2f p2Raw = detectColor(hsv, player2ColorLower, player2ColorUpper, p2Area);
        cv::Point2f p2Pos = player2Filter.update(p2Raw, p2Raw.x > 0);

        currentData.player2X = p2Pos.x / frame.cols;
        currentData.player2Y = p2Pos.y / frame.rows;
        currentData.player2Tracking = player2Filter.initialized;

        // 간단한 제스처 감지 (예시)
        static cv::Point2f lastMousePos = mousePos;
        float mouseDist = cv::norm(mousePos - lastMousePos);
        if (mouseDist > 50) {
            currentData.gesture = "SWIPE";
        }
        else {
            currentData.gesture = "NONE";
        }
        lastMousePos = mousePos;

        // UDP로 전송
        udpSender.sendData(currentData);
    }

    void drawDebug(cv::Mat& frame) {
        // 마우스
        if (currentData.mouseTracking) {
            cv::Point mousePoint(currentData.mouseX * frame.cols, currentData.mouseY * frame.rows);
            cv::circle(frame, mousePoint, 20, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, currentData.mouseClick ? "CLICK" : "MOUSE",
                mousePoint + cv::Point(-20, -25), cv::FONT_HERSHEY_SIMPLEX,
                0.5, cv::Scalar(0, 255, 0), 1);
        }

        // Player 1
        if (currentData.player1Tracking) {
            cv::Point p1Point(currentData.player1X * frame.cols, currentData.player1Y * frame.rows);
            cv::circle(frame, p1Point, 25, cv::Scalar(0, 0, 255), 2);
            cv::putText(frame, "P1", p1Point + cv::Point(-10, 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        }

        // Player 2
        if (currentData.player2Tracking) {
            cv::Point p2Point(currentData.player2X * frame.cols, currentData.player2Y * frame.rows);
            cv::circle(frame, p2Point, 25, cv::Scalar(0, 255, 255), 2);
            cv::putText(frame, "P2", p2Point + cv::Point(-10, 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        }

        // 상태 표시
        cv::putText(frame, "UDP -> UE5.4 | Port: 7777", cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

        std::stringstream status;
        status << "Time: " << std::fixed << std::setprecision(1) << currentData.timestamp
            << "s | FPS: " << cv::getTickFrequency() / cv::getTickCount();
        cv::putText(frame, status.str(), cv::Point(10, 60),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);
    }
};

int main() {
    std::cout << "=== OpenCV to Unreal Engine 5.4 UDP Sender ===" << std::endl;
    std::cout << "Sending JSON data to localhost:7777" << std::endl;
    std::cout << "Green: Mouse | Red: Player1 | Yellow: Player2" << std::endl << std::endl;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera!" << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS, 60);

    OpenCVTracker tracker;

    cv::namedWindow("OpenCV Tracking", cv::WINDOW_NORMAL);

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) continue;

        cv::flip(frame, frame, 1);

        tracker.processFrame(frame);
        tracker.drawDebug(frame);

        cv::imshow("OpenCV Tracking", frame);

        char key = cv::waitKey(1);
        if (key == 27) break;  // ESC

        // 60 FPS 유지
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}