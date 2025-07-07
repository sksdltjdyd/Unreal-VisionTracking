// color_ball_tracker.cpp
// ���� ��� �ǽð� �� ���� ���α׷�

#include <opencv2/opencv.hpp>
#include <iostream>
#include <deque>

class ColorBallTracker {
private:
    // HSV ���� ���� (�⺻��: ��Ȳ�� ��)
    cv::Scalar lowerBound;
    cv::Scalar upperBound;

    // ���� ����
    std::deque<cv::Point> trajectory;
    const int maxTrajectorySize = 50;

    // ������ ���͸��� ���� �ּ� ũ��
    const double minContourArea = 500.0;

    // Į�� ���� (�ε巯�� ����)
    cv::KalmanFilter kalman;
    cv::Mat measurement;
    cv::Mat prediction;
    bool isKalmanInitialized = false;

public:
    ColorBallTracker() {
        // ��Ȳ�� �� ���� HSV ����
        // H: 5-25, S: 100-255, V: 100-255
        setColorRange(cv::Scalar(5, 100, 100), cv::Scalar(25, 255, 255));

        // Į�� ���� �ʱ�ȭ (2D ��ġ ����)
        kalman.init(4, 2, 0); // ����: [x, y, vx, vy], ����: [x, y]
        measurement = cv::Mat::zeros(2, 1, CV_32F);

        // ���� ��� (��� � ��)
        kalman.transitionMatrix = (cv::Mat_<float>(4, 4) <<
            1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1);

        // ���� ���
        kalman.measurementMatrix = cv::Mat::eye(2, 4, CV_32F);

        // ���μ��� ������
        cv::setIdentity(kalman.processNoiseCov, cv::Scalar::all(0.1));

        // ���� ������
        cv::setIdentity(kalman.measurementNoiseCov, cv::Scalar::all(10));

        // �ʱ� ����
        cv::setIdentity(kalman.errorCovPost, cv::Scalar::all(1));
    }

    void setColorRange(const cv::Scalar& lower, const cv::Scalar& upper) {
        lowerBound = lower;
        upperBound = upper;
    }

    cv::Point trackBall(const cv::Mat& frame, cv::Mat& debugFrame) {
        cv::Mat hsv, mask;

        // BGR�� HSV�� ��ȯ
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        // ���� ������ ����ũ ����
        cv::inRange(hsv, lowerBound, upperBound, mask);

        // ������ ���� (�������� ����)
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

        // ����þ� ���� �߰� ������ ����
        cv::GaussianBlur(mask, mask, cv::Size(9, 9), 2, 2);

        // ������ ã��
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Point ballCenter(-1, -1);

        if (!contours.empty()) {
            // ���� ū ������ ã��
            auto largestContour = std::max_element(contours.begin(), contours.end(),
                [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                    return cv::contourArea(a) < cv::contourArea(b);
                });

            double area = cv::contourArea(*largestContour);

            if (area > minContourArea) {
                // �ּ� ������ ���
                cv::Point2f center;
                float radius;
                cv::minEnclosingCircle(*largestContour, center, radius);

                ballCenter = cv::Point(center.x, center.y);

                // Į�� ���� ������Ʈ
                if (!isKalmanInitialized) {
                    // ù ���� �� Į�� ���� �ʱ�ȭ
                    kalman.statePost = (cv::Mat_<float>(4, 1) <<
                        ballCenter.x, ballCenter.y, 0, 0);
                    isKalmanInitialized = true;
                }

                // ������ ������Ʈ
                measurement.at<float>(0) = ballCenter.x;
                measurement.at<float>(1) = ballCenter.y;

                // Į�� ���ͷ� ����
                cv::Mat corrected = kalman.correct(measurement);
                ballCenter.x = corrected.at<float>(0);
                ballCenter.y = corrected.at<float>(1);

                // ����� ȭ�鿡 ���� ǥ��
                if (!debugFrame.empty()) {
                    // ���� ������ �׸���
                    cv::drawContours(debugFrame, contours,
                        std::distance(contours.begin(), largestContour),
                        cv::Scalar(0, 255, 0), 2);

                    // ������ �� �׸���
                    cv::circle(debugFrame, ballCenter, radius,
                        cv::Scalar(0, 255, 255), 2);

                    // �߽��� ǥ��
                    cv::circle(debugFrame, ballCenter, 5,
                        cv::Scalar(0, 0, 255), -1);

                    // ���� �ؽ�Ʈ
                    std::string info = "Ball: (" + std::to_string(ballCenter.x) +
                        ", " + std::to_string(ballCenter.y) +
                        ") R:" + std::to_string((int)radius);
                    cv::putText(debugFrame, info, cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                }

                // ���� ������Ʈ
                trajectory.push_back(ballCenter);
                if (trajectory.size() > maxTrajectorySize) {
                    trajectory.pop_front();
                }
            }
        }
        else if (isKalmanInitialized) {
            // ���� �� ã���� ���� ������ ���
            prediction = kalman.predict();
            ballCenter.x = prediction.at<float>(0);
            ballCenter.y = prediction.at<float>(1);

            if (!debugFrame.empty()) {
                cv::circle(debugFrame, ballCenter, 20,
                    cv::Scalar(255, 0, 0), 2); // �Ķ���: ���� ��ġ
                cv::putText(debugFrame, "Predicted", ballCenter + cv::Point(10, -10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
            }
        }

        // ���� �׸���
        if (!debugFrame.empty() && trajectory.size() > 1) {
            for (size_t i = 1; i < trajectory.size(); i++) {
                int thickness = (int)(i * 3.0 / trajectory.size()) + 1;
                cv::line(debugFrame, trajectory[i - 1], trajectory[i],
                    cv::Scalar(0, 165, 255), thickness);
            }
        }

        // ����ũ ǥ�� (���� â����)
        if (!mask.empty()) {
            cv::Mat smallMask;
            cv::resize(mask, smallMask, cv::Size(320, 240));
            cv::imshow("Color Mask", smallMask);
        }

        return ballCenter;
    }

    void clearTrajectory() {
        trajectory.clear();
    }
};

// HSV ���� ������ ���� Ʈ���� �ݹ�
cv::Mat* g_frame = nullptr;
ColorBallTracker* g_tracker = nullptr;

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