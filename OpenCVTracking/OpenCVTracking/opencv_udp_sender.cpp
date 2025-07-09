// opencv_to_unreal_udp_fixed.cpp
// OpenCV tracking data to Unreal Engine via UDP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <chrono>
#include <memory>
#include <deque>

// Platform specific includes
#ifdef _WIN32
#define _WINSOCK_DEPRECATED_NO_WARNINGS  // inet_addr warning disable
#include <winsock2.h>
#include <ws2tcpip.h>  // For inet_pton
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

// ===== UDP Sender Class =====
class SimpleUDPSender {
private:
    SOCKET sock;  // Use SOCKET type for Windows compatibility
    struct sockaddr_in server_addr;
    bool initialized;

public:
    SimpleUDPSender() : sock(INVALID_SOCKET), initialized(false) {
        // Initialize all members
        memset(&server_addr, 0, sizeof(server_addr));
    }

    bool init(const std::string& ip = "127.0.0.1", int port = 7777) {
#ifdef _WIN32
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            std::cerr << "Winsock initialization failed!" << std::endl;
            return false;
        }
#endif

        // Create UDP socket
        sock = socket(AF_INET, SOCK_DGRAM, 0);
        if (sock == INVALID_SOCKET) {
            std::cerr << "Socket creation failed!" << std::endl;
            return false;
        }

        // Configure server address
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(static_cast<u_short>(port));

        // Use inet_pton instead of deprecated inet_addr
#ifdef _WIN32
        if (inet_pton(AF_INET, ip.c_str(), &server_addr.sin_addr) != 1) {
            std::cerr << "Invalid IP address!" << std::endl;
            closesocket(sock);
            return false;
        }
#else
        if (inet_pton(AF_INET, ip.c_str(), &server_addr.sin_addr) != 1) {
            std::cerr << "Invalid IP address!" << std::endl;
            close(sock);
            return false;
        }
#endif

        initialized = true;
        std::cout << "UDP Sender ready: " << ip << ":" << port << std::endl;
        return true;
    }

    void sendTrackingData(
        float mouseX, float mouseY, bool mouseClick,
        float p1X, float p1Y, bool p1Track,
        float p2X, float p2Y, bool p2Track,
        int p1Score, int p2Score,
        const std::string& gesture
    ) {
        if (!initialized) return;

        std::stringstream ss;
        ss << "TRACK|";
        ss << mouseX << "|" << mouseY << "|" << (mouseClick ? 1 : 0) << "|";
        ss << p1X << "|" << p1Y << "|" << (p1Track ? 1 : 0) << "|";
        ss << p2X << "|" << p2Y << "|" << (p2Track ? 1 : 0) << "|";
        ss << p1Score << "|" << p2Score << "|";
        ss << gesture;

        std::string data = ss.str();

        // Send data - use static_cast to avoid conversion warnings
        int result = sendto(sock, data.c_str(), static_cast<int>(data.length()), 0,
            reinterpret_cast<struct sockaddr*>(&server_addr),
            static_cast<int>(sizeof(server_addr)));

        if (result == SOCKET_ERROR) {
#ifdef _WIN32
            std::cerr << "Send failed: " << WSAGetLastError() << std::endl;
#else
            std::cerr << "Send failed" << std::endl;
#endif
        }
    }

    ~SimpleUDPSender() {
        if (initialized && sock != INVALID_SOCKET) {
#ifdef _WIN32
            closesocket(sock);
            WSACleanup();
#else
            close(sock);
#endif
        }
    }
};

// ===== Kalman Filter Tracker =====
class KalmanTracker {
private:
    cv::KalmanFilter kf;
    cv::Mat state;
    cv::Mat measurement;
    bool initialized;

public:
    cv::Point2f lastPosition;
    bool isTracking;

    KalmanTracker() : initialized(false), isTracking(false), lastPosition(0, 0) {
        kf.init(4, 2, 0);

        kf.transitionMatrix = (cv::Mat_<float>(4, 4) <<
            1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1);

        kf.measurementMatrix = (cv::Mat_<float>(2, 4) <<
            1, 0, 0, 0,
            0, 1, 0, 0);

        cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-3));
        cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-2));
        cv::setIdentity(kf.errorCovPost, cv::Scalar::all(.1));

        state = cv::Mat::zeros(4, 1, CV_32F);
        measurement = cv::Mat::zeros(2, 1, CV_32F);
    }

    cv::Point2f update(const cv::Point2f& measuredPt, bool found) {
        if (!initialized && found) {
            state.at<float>(0) = measuredPt.x;
            state.at<float>(1) = measuredPt.y;
            state.at<float>(2) = 0;
            state.at<float>(3) = 0;

            kf.statePre = state.clone();
            kf.statePost = state.clone();

            initialized = true;
            isTracking = true;
            lastPosition = measuredPt;
            return measuredPt;
        }

        if (!initialized) {
            isTracking = false;
            return cv::Point2f(-1, -1);
        }

        cv::Mat prediction = kf.predict();

        if (found) {
            measurement.at<float>(0) = measuredPt.x;
            measurement.at<float>(1) = measuredPt.y;
            cv::Mat corrected = kf.correct(measurement);

            lastPosition.x = corrected.at<float>(0);
            lastPosition.y = corrected.at<float>(1);
            isTracking = true;
        }
        else {
            lastPosition.x = prediction.at<float>(0);
            lastPosition.y = prediction.at<float>(1);
        }

        return lastPosition;
    }
};

// ===== Simple Tracking Game with UDP =====
class SimplePongWithUDP {
private:
    const int GAME_WIDTH = 1280;
    const int GAME_HEIGHT = 720;

    std::unique_ptr<KalmanTracker> mouseTracker;
    std::unique_ptr<KalmanTracker> player1Tracker;
    std::unique_ptr<KalmanTracker> player2Tracker;

    SimpleUDPSender udpSender;

    std::deque<cv::Point2f> gesturePoints;

public:
    // Game data structure with proper initialization
    struct GameData {
        cv::Point2f mousePos;
        bool mouseClick;
        cv::Point2f player1Pos;
        cv::Point2f player2Pos;
        int player1Score;
        int player2Score;
        std::string currentGesture;

        // Constructor to initialize all members
        GameData() : mousePos(0, 0), mouseClick(false),
            player1Pos(0, 0), player2Pos(0, 0),
            player1Score(0), player2Score(0),
            currentGesture("NONE") {
        }
    } gameData;

    // HSV ranges
    int mouse_h_min, mouse_h_max;
    int mouse_s_min, mouse_s_max;
    int mouse_v_min, mouse_v_max;

    int p1_h_min, p1_h_max;
    int p1_s_min, p1_s_max;
    int p1_v_min, p1_v_max;

    int p2_h_min, p2_h_max;
    int p2_s_min, p2_s_max;
    int p2_v_min, p2_v_max;

    SimplePongWithUDP() :
        // Initialize all HSV values
        mouse_h_min(40), mouse_h_max(80),
        mouse_s_min(100), mouse_s_max(255),
        mouse_v_min(100), mouse_v_max(255),
        p1_h_min(0), p1_h_max(10),
        p1_s_min(120), p1_s_max(255),
        p1_v_min(100), p1_v_max(255),
        p2_h_min(20), p2_h_max(40),
        p2_s_min(120), p2_s_max(255),
        p2_v_min(100), p2_v_max(255) {

        mouseTracker = std::make_unique<KalmanTracker>();
        player1Tracker = std::make_unique<KalmanTracker>();
        player2Tracker = std::make_unique<KalmanTracker>();

        udpSender.init("127.0.0.1", 7777);
    }

    cv::Point2f detectColor(const cv::Mat& hsv, cv::Scalar lower, cv::Scalar upper, cv::Mat& mask) {
        cv::inRange(hsv, lower, upper, mask);

        cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), 2);
        cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), 2);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if (!contours.empty()) {
            double maxArea = 0;
            int maxIdx = -1;
            for (size_t i = 0; i < contours.size(); i++) {
                double area = cv::contourArea(contours[i]);
                if (area > maxArea && area > 500) {
                    maxArea = area;
                    maxIdx = static_cast<int>(i);
                }
            }

            if (maxIdx >= 0) {
                cv::Moments m = cv::moments(contours[maxIdx]);
                if (m.m00 != 0) {
                    return cv::Point2f(static_cast<float>(m.m10 / m.m00),
                        static_cast<float>(m.m01 / m.m00));
                }
            }
        }

        return cv::Point2f(-1, -1);
    }

    std::string detectGesture() {
        if (gesturePoints.size() < 20) return "NONE";

        float totalDx = 0, totalDy = 0;
        for (size_t i = 1; i < gesturePoints.size(); i++) {
            totalDx += std::abs(gesturePoints[i].x - gesturePoints[i - 1].x);
            totalDy += std::abs(gesturePoints[i].y - gesturePoints[i - 1].y);
        }

        if (totalDy > totalDx * 2 && totalDy > 100) return "VERTICAL";
        if (totalDx > totalDy * 2 && totalDx > 100) return "HORIZONTAL";

        float dist = static_cast<float>(cv::norm(gesturePoints.front() - gesturePoints.back()));
        if (dist < 50 && totalDx + totalDy > 300) return "CIRCLE";

        return "NONE";
    }

    void update(const cv::Mat& frame) {
        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        // Mouse tracking (green)
        cv::Mat mouseMask;
        cv::Point2f mouseRaw = detectColor(hsv,
            cv::Scalar(mouse_h_min, mouse_s_min, mouse_v_min),
            cv::Scalar(mouse_h_max, mouse_s_max, mouse_v_max),
            mouseMask);

        gameData.mousePos = mouseTracker->update(mouseRaw, mouseRaw.x > 0);

        int mouseArea = cv::countNonZero(mouseMask);
        gameData.mouseClick = (mouseArea > 0 && mouseArea < 2000);

        // Player 1 tracking (red)
        cv::Mat p1Mask;
        cv::Point2f p1Raw = detectColor(hsv,
            cv::Scalar(p1_h_min, p1_s_min, p1_v_min),
            cv::Scalar(p1_h_max, p1_s_max, p1_v_max),
            p1Mask);

        gameData.player1Pos = player1Tracker->update(p1Raw, p1Raw.x > 0);

        // Player 2 tracking (yellow)
        cv::Mat p2Mask;
        cv::Point2f p2Raw = detectColor(hsv,
            cv::Scalar(p2_h_min, p2_s_min, p2_v_min),
            cv::Scalar(p2_h_max, p2_s_max, p2_v_max),
            p2Mask);

        gameData.player2Pos = player2Tracker->update(p2Raw, p2Raw.x > 0);

        // Gesture detection
        if (mouseTracker->isTracking) {
            gesturePoints.push_back(gameData.mousePos);
            if (gesturePoints.size() > 30) gesturePoints.pop_front();

            gameData.currentGesture = detectGesture();
        }

        // Send UDP data
        udpSender.sendTrackingData(
            gameData.mousePos.x / GAME_WIDTH,
            gameData.mousePos.y / GAME_HEIGHT,
            gameData.mouseClick,
            gameData.player1Pos.x / GAME_WIDTH,
            gameData.player1Pos.y / GAME_HEIGHT,
            player1Tracker->isTracking,
            gameData.player2Pos.x / GAME_WIDTH,
            gameData.player2Pos.y / GAME_HEIGHT,
            player2Tracker->isTracking,
            gameData.player1Score,
            gameData.player2Score,
            gameData.currentGesture
        );
    }

    void draw(cv::Mat& frame) {
        // Draw mouse cursor
        if (mouseTracker->isTracking) {
            cv::circle(frame, cv::Point(static_cast<int>(gameData.mousePos.x),
                static_cast<int>(gameData.mousePos.y)),
                20, gameData.mouseClick ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 200, 0), 3);

            for (size_t i = 1; i < gesturePoints.size(); i++) {
                cv::line(frame,
                    cv::Point(static_cast<int>(gesturePoints[i - 1].x),
                        static_cast<int>(gesturePoints[i - 1].y)),
                    cv::Point(static_cast<int>(gesturePoints[i].x),
                        static_cast<int>(gesturePoints[i].y)),
                    cv::Scalar(0, 255, 0), 2);
            }
        }

        // Draw Player 1
        if (player1Tracker->isTracking) {
            cv::circle(frame, cv::Point(static_cast<int>(gameData.player1Pos.x),
                static_cast<int>(gameData.player1Pos.y)),
                30, cv::Scalar(0, 0, 255), 3);
            cv::putText(frame, "P1",
                cv::Point(static_cast<int>(gameData.player1Pos.x - 15),
                    static_cast<int>(gameData.player1Pos.y + 5)),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        }

        // Draw Player 2
        if (player2Tracker->isTracking) {
            cv::circle(frame, cv::Point(static_cast<int>(gameData.player2Pos.x),
                static_cast<int>(gameData.player2Pos.y)),
                30, cv::Scalar(0, 255, 255), 3);
            cv::putText(frame, "P2",
                cv::Point(static_cast<int>(gameData.player2Pos.x - 15),
                    static_cast<int>(gameData.player2Pos.y + 5)),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        }

        // Draw info
        cv::putText(frame, "UDP -> Unreal Engine", cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        cv::putText(frame, "Gesture: " + gameData.currentGesture, cv::Point(10, 60),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);

        std::stringstream status;
        status << "Mouse: " << (mouseTracker->isTracking ? "OK" : "NO")
            << " | P1: " << (player1Tracker->isTracking ? "OK" : "NO")
            << " | P2: " << (player2Tracker->isTracking ? "OK" : "NO");
        cv::putText(frame, status.str(), cv::Point(10, 90),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);
    }
};

// ===== Main Function =====
int main() {
    std::cout << "=== OpenCV to Unreal UDP Bridge ===" << std::endl;
    std::cout << "Green: Mouse cursor (make fist to click)" << std::endl;
    std::cout << "Red: Player 1" << std::endl;
    std::cout << "Yellow: Player 2" << std::endl;
    std::cout << "Data is sent to localhost:7777" << std::endl << std::endl;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open webcam!" << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    SimplePongWithUDP game;

    cv::namedWindow("OpenCV to Unreal", cv::WINDOW_NORMAL);
    cv::namedWindow("HSV Controls", cv::WINDOW_NORMAL);

    // Create trackbars
    cv::createTrackbar("Mouse H Min", "HSV Controls", &game.mouse_h_min, 180);
    cv::createTrackbar("Mouse H Max", "HSV Controls", &game.mouse_h_max, 180);
    cv::createTrackbar("Mouse S Min", "HSV Controls", &game.mouse_s_min, 255);
    cv::createTrackbar("Mouse S Max", "HSV Controls", &game.mouse_s_max, 255);
    cv::createTrackbar("Mouse V Min", "HSV Controls", &game.mouse_v_min, 255);
    cv::createTrackbar("Mouse V Max", "HSV Controls", &game.mouse_v_max, 255);

    cv::createTrackbar("P1 H Min", "HSV Controls", &game.p1_h_min, 180);
    cv::createTrackbar("P1 H Max", "HSV Controls", &game.p1_h_max, 180);
    cv::createTrackbar("P1 S Min", "HSV Controls", &game.p1_s_min, 255);
    cv::createTrackbar("P1 S Max", "HSV Controls", &game.p1_s_max, 255);
    cv::createTrackbar("P1 V Min", "HSV Controls", &game.p1_v_min, 255);
    cv::createTrackbar("P1 V Max", "HSV Controls", &game.p1_v_max, 255);

    cv::createTrackbar("P2 H Min", "HSV Controls", &game.p2_h_min, 180);
    cv::createTrackbar("P2 H Max", "HSV Controls", &game.p2_h_max, 180);
    cv::createTrackbar("P2 S Min", "HSV Controls", &game.p2_s_min, 255);
    cv::createTrackbar("P2 S Max", "HSV Controls", &game.p2_s_max, 255);
    cv::createTrackbar("P2 V Min", "HSV Controls", &game.p2_v_min, 255);
    cv::createTrackbar("P2 V Max", "HSV Controls", &game.p2_v_max, 255);

    auto lastTime = std::chrono::high_resolution_clock::now();
    float fps = 0;

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) continue;

        cv::flip(frame, frame, 1);

        game.update(frame);
        game.draw(frame);

        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - lastTime).count();
        fps = 0.9f * fps + 0.1f * (1.0f / dt);
        lastTime = now;

        cv::putText(frame, "FPS: " + std::to_string(static_cast<int>(fps)),
            cv::Point(10, 120),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);

        cv::imshow("OpenCV to Unreal", frame);

        char key = cv::waitKey(1);
        if (key == 27) break;

        if (key == 'd' || key == 'D') {
            std::cout << "Current data:" << std::endl;
            std::cout << "Mouse: (" << game.gameData.mousePos.x << ", "
                << game.gameData.mousePos.y << ") Click: "
                << game.gameData.mouseClick << std::endl;
            std::cout << "P1: (" << game.gameData.player1Pos.x << ", "
                << game.gameData.player1Pos.y << ")" << std::endl;
            std::cout << "P2: (" << game.gameData.player2Pos.x << ", "
                << game.gameData.player2Pos.y << ")" << std::endl;
            std::cout << "Gesture: " << game.gameData.currentGesture << std::endl;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}