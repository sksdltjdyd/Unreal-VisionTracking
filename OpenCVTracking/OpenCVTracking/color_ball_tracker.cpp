// color_ball_tracker.cpp
// ���� ��� �ǽð� �� ���� ���α׷�

#include "mainheader.h"
#include "opencv.h"
#include "osc.h"

// HSV ���� ������ ���� Ʈ���� �ݹ�
cv::Mat* g_frame = nullptr;
ColorBallTracker* g_tracker = nullptr;

int main() {
    std::cout << "=== ���� ��� �� ���� ���α׷� ===" << std::endl;
    std::cout << "����:" << std::endl;
    std::cout << "- ESC: ����" << std::endl;
    std::cout << "- SPACE: ���� �ʱ�ȭ" << std::endl;
    std::cout << "- Ʈ���ٷ� ���� ���� ����" << std::endl;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "��ķ�� �� �� �����ϴ�!" << std::endl;
        return -1;
    }

    // �ػ� ����
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS, 60);

    ColorBallTracker tracker;
    g_tracker = &tracker;

    cv::namedWindow("Ball Tracking", cv::WINDOW_NORMAL);
    cv::namedWindow("Controls", cv::WINDOW_NORMAL);
    cv::resizeWindow("Controls", 400, 300);

    // HSV ���� Ʈ����
    cv::createTrackbar("H Low", "Controls", nullptr, 180, on_trackbar);
    cv::createTrackbar("H High", "Controls", nullptr, 180, on_trackbar);
    cv::createTrackbar("S Low", "Controls", nullptr, 255, on_trackbar);
    cv::createTrackbar("S High", "Controls", nullptr, 255, on_trackbar);
    cv::createTrackbar("V Low", "Controls", nullptr, 255, on_trackbar);
    cv::createTrackbar("V High", "Controls", nullptr, 255, on_trackbar);

    // �ʱⰪ ���� (��Ȳ��)
    cv::setTrackbarPos("H Low", "Controls", 20);
    cv::setTrackbarPos("H High", "Controls", 25);
    cv::setTrackbarPos("S Low", "Controls", 100);
    cv::setTrackbarPos("S High", "Controls", 255);
    cv::setTrackbarPos("V Low", "Controls", 100);
    cv::setTrackbarPos("V High", "Controls", 255);

    cv::Mat frame;
    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    double fps = 0.0;

    while (true) {
        cap >> frame;
        if (frame.empty()) continue;

        g_frame = &frame;
        cv::Mat displayFrame = frame.clone();

        // �� ����
        cv::Point ballPos = tracker.trackBall(frame, displayFrame);

        // FPS ���
        frame_count++;
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - start_time).count();

        if (elapsed > 1000) {
            fps = frame_count * 1000.0 / elapsed;
            frame_count = 0;
            start_time = current_time;
        }

        // FPS ǥ��
        cv::putText(displayFrame, "FPS: " + std::to_string((int)fps),
            cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX,
            0.7, cv::Scalar(255, 255, 0), 2);

        cv::imshow("Ball Tracking", displayFrame);

        char key = cv::waitKey(1);
        if (key == 27) break;  // ESC
        else if (key == ' ') tracker.clearTrajectory();  // SPACE
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}

void on_trackbar(int, void*) {
    // Ʈ���� ������ ���� ���� ������Ʈ
    int h_low = cv::getTrackbarPos("H Low", "Controls");
    int h_high = cv::getTrackbarPos("H High", "Controls");
    int s_low = cv::getTrackbarPos("S Low", "Controls");
    int s_high = cv::getTrackbarPos("S High", "Controls");
    int v_low = cv::getTrackbarPos("V Low", "Controls");
    int v_high = cv::getTrackbarPos("V High", "Controls");

    if (g_tracker) {
        g_tracker->setColorRange(
            cv::Scalar(h_low, s_low, v_low),
            cv::Scalar(h_high, s_high, v_high)
        );
    }
}
