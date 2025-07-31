// OpenCV OSC Tracking System
// ���� ���� �����͸� OSC�� �𸮾� ������ ����

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <deque>
#include <memory>

// OSCPack ���̺귯�� ����
#include "C:/Libraries/oscpack/osc/OscOutboundPacketStream.h"
#include "C:/Libraries/oscpack/ip/UdpSocket.h"

// ��� ����
const int OUTPUT_BUFFER_SIZE = 4096;
const int CAMERA_WIDTH = 1280;
const int CAMERA_HEIGHT = 720;

// ===== OSC �۽��� Ŭ���� =====
class OSCSender {
private:
    UdpTransmitSocket* transmitSocket;
    char buffer[OUTPUT_BUFFER_SIZE];

public:
    // ������ - OSC ���� ���� ����
    OSCSender(const std::string& address = "127.0.0.1", int port = 8000) {
        try {
            // UDP ������ ���� OSC �޽��� ����
            transmitSocket = new UdpTransmitSocket(
                IpEndpointName(address.c_str(), port)
            );
            std::cout << "OSC Sender initialized: " << address << ":" << port << std::endl;
        }
        catch (std::exception& e) {
            std::cerr << "OSC �ʱ�ȭ ����: " << e.what() << std::endl;
            transmitSocket = nullptr;
        }
    }

    // �Ҹ���
    ~OSCSender() {
        if (transmitSocket) {
            delete transmitSocket;
        }
    }

    // ��� Ʈ��ŷ �����͸� �� ���� ����
    void sendAllTrackingData(
        float mouseX, float mouseY, bool mouseClick,
        float p1X, float p1Y, bool p1Tracking,
        float p2X, float p2Y, bool p2Tracking) {

        if (!transmitSocket) return;

        // OSC ��Ŷ ��Ʈ�� ����
        osc::OutboundPacketStream packet(buffer, OUTPUT_BUFFER_SIZE);

        try {
            // OSC �޽��� ����
            // �ּ� ����: /tracking/all
            // ������: mouseX, mouseY, mouseClick, p1X, p1Y, p2X, p2Y, p1Track, p2Track
            packet << osc::BeginMessage("/tracking/all")
                << mouseX << mouseY << (mouseClick ? 1 : 0)    // ���콺 ������
                << p1X << p1Y                                  // Player 1 ��ġ
                << p2X << p2Y                                  // Player 2 ��ġ
                << (p1Tracking ? 1 : 0)                        // Player 1 ���� ����
                << (p2Tracking ? 1 : 0)                        // Player 2 ���� ����
                << osc::EndMessage;

            // ��Ŷ ����
            transmitSocket->Send(packet.Data(), packet.Size());

        }
        catch (std::exception& e) {
            std::cerr << "OSC ���� ����: " << e.what() << std::endl;
        }
    }

    // ���� ������ ���� �޼���� (������ ���)
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

    // ����ó ����
    void sendGesture(const std::string& gesture) {
        if (!transmitSocket) return;

        osc::OutboundPacketStream packet(buffer, OUTPUT_BUFFER_SIZE);
        packet << osc::BeginMessage("/tracking/gesture")
            << gesture.c_str()
            << osc::EndMessage;

        transmitSocket->Send(packet.Data(), packet.Size());
    }
};

// ===== ������ Į�� ���� (�ε巯�� ������) =====
class SimpleKalmanFilter {
private:
    cv::Point2f position;
    cv::Point2f velocity;
    bool initialized;
    float alpha;  // ������ ����

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
            // ���ο� �������� ���� ��
            cv::Point2f newPos = measurement;
            velocity = newPos - position;

            // ������ ���� (Low-pass filter)
            position = position * alpha + newPos * (1.0f - alpha);
        }
        else {
            // �������� ���� ���� ����
            position += velocity * 0.9f;  // �ӵ� ���� ����
        }

        return position;
    }

    bool isTracking() const { return initialized; }
    void reset() { initialized = false; velocity = cv::Point2f(0, 0); }
};

// ===== ���� Ʈ��ŷ �ý��� =====
class OSCTrackingSystem {
private:
    // OSC �۽���
    OSCSender oscSender;

    // Į�� ���͵� (�� ���� ���)
    SimpleKalmanFilter mouseFilter;
    SimpleKalmanFilter player1Filter;
    SimpleKalmanFilter player2Filter;

    // ����ó �νĿ�
    std::deque<cv::Point2f> gesturePoints;

    // HSV ���� ���� ����
    struct ColorRange {
        cv::Scalar lower;
        cv::Scalar upper;
        std::string name;
    };

    ColorRange mouseColor;    // �ʷϻ�
    ColorRange player1Color;  // ������
    ColorRange player2Color;  // �����

    // FPS ����
    std::chrono::steady_clock::time_point lastTime;
    float fps;

public:
    // ������ - �⺻ ���� ����
    OSCTrackingSystem() : oscSender("127.0.0.1", 8000), fps(0) {
        // �ʷϻ� ���콺
        mouseColor.lower = cv::Scalar(40, 100, 100);
        mouseColor.upper = cv::Scalar(80, 255, 255);
        mouseColor.name = "Mouse (Green)";

        // ������ Player 1
        player1Color.lower = cv::Scalar(0, 120, 100);
        player1Color.upper = cv::Scalar(10, 255, 255);
        player1Color.name = "Player 1 (Red)";

        // ����� Player 2
        player2Color.lower = cv::Scalar(20, 120, 100);
        player2Color.upper = cv::Scalar(40, 255, 255);
        player2Color.name = "Player 2 (Yellow)";

        lastTime = std::chrono::steady_clock::now();
    }

    // ���� ���� �Լ�
    cv::Point2f detectColor(const cv::Mat& hsv, const ColorRange& color,
        cv::Mat& outputMask, int& area) {
        // ���� ������ ����ũ ����
        cv::inRange(hsv, color.lower, color.upper, outputMask);

        // ������ ���� (�������� ����)
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(outputMask, outputMask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(outputMask, outputMask, cv::MORPH_CLOSE, kernel);

        // ������ ã��
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(outputMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // ���� ū ������ ã��
        double maxArea = 0;
        int maxIdx = -1;

        for (size_t i = 0; i < contours.size(); i++) {
            double currentArea = cv::contourArea(contours[i]);
            if (currentArea > maxArea && currentArea > 500) {  // �ּ� ũ�� ����
                maxArea = currentArea;
                maxIdx = static_cast<int>(i);
            }
        }

        // �߽��� ���
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
        return cv::Point2f(-1, -1);  // ���� ����
    }

    // ������ ����ó �ν�
    std::string detectGesture() {
        if (gesturePoints.size() < 20) return "NONE";

        // ��ü �̵� �Ÿ� ���
        float totalDistance = 0;
        float totalDx = 0, totalDy = 0;

        for (size_t i = 1; i < gesturePoints.size(); i++) {
            cv::Point2f diff = gesturePoints[i] - gesturePoints[i - 1];
            totalDistance += cv::norm(diff);
            totalDx += std::abs(diff.x);
            totalDy += std::abs(diff.y);
        }

        // ����ó �Ǻ�
        if (totalDistance < 50) return "NONE";  // �������� �ʹ� ����

        // ����/���� ������ �Ǻ�
        if (totalDy > totalDx * 2 && totalDy > 100) return "VERTICAL";
        if (totalDx > totalDy * 2 && totalDx > 100) return "HORIZONTAL";

        // ���� ������ �Ǻ�
        cv::Point2f startEnd = gesturePoints.back() - gesturePoints.front();
        float endDistance = cv::norm(startEnd);
        if (endDistance < 50 && totalDistance > 200) return "CIRCLE";

        return "NONE";
    }

    // ���� ������ ó�� �Լ�
    void processFrame(const cv::Mat& frame) {
        // BGR�� HSV�� ��ȯ (���� ������ �� ��Ȯ��)
        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        // 1. ���콺(�ʷϻ�) ����
        cv::Mat mouseMask;
        int mouseArea;
        cv::Point2f mouseRaw = detectColor(hsv, mouseColor, mouseMask, mouseArea);
        cv::Point2f mousePos = mouseFilter.update(mouseRaw, mouseRaw.x > 0);

        // Ŭ�� ���� (�ָ��� ��� ������ �پ��)
        bool mouseClick = (mouseArea > 0 && mouseArea < 2000);

        // 2. Player 1 (������) ����
        cv::Mat p1Mask;
        int p1Area;
        cv::Point2f p1Raw = detectColor(hsv, player1Color, p1Mask, p1Area);
        cv::Point2f p1Pos = player1Filter.update(p1Raw, p1Raw.x > 0);

        // 3. Player 2 (�����) ����
        cv::Mat p2Mask;
        int p2Area;
        cv::Point2f p2Raw = detectColor(hsv, player2Color, p2Mask, p2Area);
        cv::Point2f p2Pos = player2Filter.update(p2Raw, p2Raw.x > 0);

        // 4. ����ó ���� (���콺 ����)
        if (mouseFilter.isTracking()) {
            gesturePoints.push_back(mousePos);
            if (gesturePoints.size() > 30) {
                gesturePoints.pop_front();
            }
        }

        // 5. ��ǥ ����ȭ (0~1 ������)
        float normMouseX = mousePos.x / CAMERA_WIDTH;
        float normMouseY = mousePos.y / CAMERA_HEIGHT;
        float normP1X = p1Pos.x / CAMERA_WIDTH;
        float normP1Y = p1Pos.y / CAMERA_HEIGHT;
        float normP2X = p2Pos.x / CAMERA_WIDTH;
        float normP2Y = p2Pos.y / CAMERA_HEIGHT;

        // 6. OSC�� ������ ����
        oscSender.sendAllTrackingData(
            normMouseX, normMouseY, mouseClick,
            normP1X, normP1Y, player1Filter.isTracking(),
            normP2X, normP2Y, player2Filter.isTracking()
        );

        // ����ó ���� �� ����
        std::string gesture = detectGesture();
        if (gesture != "NONE") {
            oscSender.sendGesture(gesture);
            gesturePoints.clear();  // ����ó �ν� �� �ʱ�ȭ
        }

        // 7. ����� �ð�ȭ
        drawDebugInfo(frame, mousePos, p1Pos, p2Pos,
            mouseClick, mouseFilter.isTracking(),
            player1Filter.isTracking(), player2Filter.isTracking());
    }

    // ����� ���� �׸���
    void drawDebugInfo(cv::Mat& frame,
        const cv::Point2f& mousePos,
        const cv::Point2f& p1Pos,
        const cv::Point2f& p2Pos,
        bool mouseClick, bool mouseTracking,
        bool p1Tracking, bool p2Tracking) {

        // FPS ���
        auto now = std::chrono::steady_clock::now();
        float deltaTime = std::chrono::duration<float>(now - lastTime).count();
        fps = 0.9f * fps + 0.1f * (1.0f / deltaTime);
        lastTime = now;

        // ���콺 �׸���
        if (mouseTracking) {
            cv::circle(frame, cv::Point(mousePos), 20,
                mouseClick ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 200, 0), 3);
            cv::putText(frame, mouseClick ? "CLICK!" : "Mouse",
                cv::Point(mousePos) + cv::Point(-20, -25),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

            // ����ó ���� �׸���
            for (size_t i = 1; i < gesturePoints.size(); i++) {
                cv::line(frame, gesturePoints[i - 1], gesturePoints[i],
                    cv::Scalar(0, 255, 0), 2);
            }
        }

        // Player 1 �׸���
        if (p1Tracking) {
            cv::circle(frame, cv::Point(p1Pos), 25, cv::Scalar(0, 0, 255), 3);
            cv::putText(frame, "P1", cv::Point(p1Pos) + cv::Point(-10, 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
        }

        // Player 2 �׸���
        if (p2Tracking) {
            cv::circle(frame, cv::Point(p2Pos), 25, cv::Scalar(0, 255, 255), 3);
            cv::putText(frame, "P2", cv::Point(p2Pos) + cv::Point(-10, 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
        }

        // ���� ���� ǥ��
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

    // HSV ������ Ʈ���� ����
    void createTrackbars() {
        cv::namedWindow("HSV Controls", cv::WINDOW_NORMAL);

        // ���콺 (�ʷϻ�) ����
        cv::createTrackbar("Mouse H Min", "HSV Controls", &mouseColor.lower[0], 180);
        cv::createTrackbar("Mouse H Max", "HSV Controls", &mouseColor.upper[0], 180);
        cv::createTrackbar("Mouse S Min", "HSV Controls", &mouseColor.lower[1], 255);
        cv::createTrackbar("Mouse S Max", "HSV Controls", &mouseColor.upper[1], 255);
        cv::createTrackbar("Mouse V Min", "HSV Controls", &mouseColor.lower[2], 255);
        cv::createTrackbar("Mouse V Max", "HSV Controls", &mouseColor.upper[2], 255);

        // Player 1 (������) ����
        cv::createTrackbar("P1 H Min", "HSV Controls", &player1Color.lower[0], 180);
        cv::createTrackbar("P1 H Max", "HSV Controls", &player1Color.upper[0], 180);
        cv::createTrackbar("P1 S Min", "HSV Controls", &player1Color.lower[1], 255);
        cv::createTrackbar("P1 S Max", "HSV Controls", &player1Color.upper[1], 255);

        // Player 2 (�����) ����
        cv::createTrackbar("P2 H Min", "HSV Controls", &player2Color.lower[0], 180);
        cv::createTrackbar("P2 H Max", "HSV Controls", &player2Color.upper[0], 180);
        cv::createTrackbar("P2 S Min", "HSV Controls", &player2Color.lower[1], 255);
        cv::createTrackbar("P2 S Max", "HSV Controls", &player2Color.upper[1], 255);
    }
};

// ===== ���� �Լ� =====
int main() {
    std::cout << "=== OpenCV OSC Tracking System ===" << std::endl;
    std::cout << "Sending to: osc://localhost:8000" << std::endl;
    std::cout << "Green: Mouse (make fist to click)" << std::endl;
    std::cout << "Red: Player 1" << std::endl;
    std::cout << "Yellow: Player 2" << std::endl;
    std::cout << "Press ESC to exit" << std::endl << std::endl;

    // ī�޶� ����
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open camera!" << std::endl;
        return -1;
    }

    // ī�޶� ����
    cap.set(cv::CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT);
    cap.set(cv::CAP_PROP_FPS, 60);

    // Ʈ��ŷ �ý��� ����
    OSCTrackingSystem tracker;
    tracker.createTrackbars();

    // ���� ������
    cv::namedWindow("OSC Tracking", cv::WINDOW_NORMAL);

    // ���� ����
    cv::Mat frame;
    while (true) {
        // ������ ĸó
        cap >> frame;
        if (frame.empty()) continue;

        // �¿� ���� (�ſ� ȿ��)
        cv::flip(frame, frame, 1);

        // ������ ó��
        tracker.processFrame(frame);

        // ȭ�� ǥ��
        cv::imshow("OSC Tracking", frame);

        // Ű �Է� ó��
        char key = cv::waitKey(1);
        if (key == 27) break;  // ESC

        // 'd' Ű�� ����� ���� ���
        if (key == 'd' || key == 'D') {
            std::cout << "Debug: Check OSC messages in Unreal Engine" << std::endl;
        }
    }

    // ����
    cap.release();
    cv::destroyAllWindows();

    std::cout << "Program ended." << std::endl;
    return 0;
}