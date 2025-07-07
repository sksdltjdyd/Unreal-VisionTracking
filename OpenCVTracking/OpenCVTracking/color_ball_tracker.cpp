// color_ball_tracker.cpp
// 색상 기반 실시간 공 추적 프로그램

#include <opencv2/opencv.hpp>
#include <iostream>
#include <deque>

class ColorBallTracker {
private:
    // HSV 색상 범위 (기본값: 주황색 공)
    cv::Scalar lowerBound;
    cv::Scalar upperBound;

    // 궤적 저장
    std::deque<cv::Point> trajectory;
    const int maxTrajectorySize = 50;

    // 노이즈 필터링을 위한 최소 크기
    const double minContourArea = 500.0;

    // 칼만 필터 (부드러운 추적)
    cv::KalmanFilter kalman;
    cv::Mat measurement;
    cv::Mat prediction;
    bool isKalmanInitialized = false;

public:
    ColorBallTracker() {
        // 주황색 공 기준 HSV 범위
        // H: 5-25, S: 100-255, V: 100-255
        setColorRange(cv::Scalar(5, 100, 100), cv::Scalar(25, 255, 255));

        // 칼만 필터 초기화 (2D 위치 추적)
        kalman.init(4, 2, 0); // 상태: [x, y, vx, vy], 측정: [x, y]
        measurement = cv::Mat::zeros(2, 1, CV_32F);

        // 전이 행렬 (등속 운동 모델)
        kalman.transitionMatrix = (cv::Mat_<float>(4, 4) <<
            1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1);

        // 측정 행렬
        kalman.measurementMatrix = cv::Mat::eye(2, 4, CV_32F);

        // 프로세스 노이즈
        cv::setIdentity(kalman.processNoiseCov, cv::Scalar::all(0.1));

        // 측정 노이즈
        cv::setIdentity(kalman.measurementNoiseCov, cv::Scalar::all(10));

        // 초기 상태
        cv::setIdentity(kalman.errorCovPost, cv::Scalar::all(1));
    }

    void setColorRange(const cv::Scalar& lower, const cv::Scalar& upper) {
        lowerBound = lower;
        upperBound = upper;
    }

    cv::Point trackBall(const cv::Mat& frame, cv::Mat& debugFrame) {
        cv::Mat hsv, mask;

        // BGR을 HSV로 변환
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        // 색상 범위로 마스크 생성
        cv::inRange(hsv, lowerBound, upperBound, mask);

        // 노이즈 제거 (모폴로지 연산)
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

        // 가우시안 블러로 추가 노이즈 제거
        cv::GaussianBlur(mask, mask, cv::Size(9, 9), 2, 2);

        // 컨투어 찾기
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Point ballCenter(-1, -1);

        if (!contours.empty()) {
            // 가장 큰 컨투어 찾기
            auto largestContour = std::max_element(contours.begin(), contours.end(),
                [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                    return cv::contourArea(a) < cv::contourArea(b);
                });

            double area = cv::contourArea(*largestContour);

            if (area > minContourArea) {
                // 최소 외접원 계산
                cv::Point2f center;
                float radius;
                cv::minEnclosingCircle(*largestContour, center, radius);

                ballCenter = cv::Point(center.x, center.y);

                // 칼만 필터 업데이트
                if (!isKalmanInitialized) {
                    // 첫 감지 시 칼만 필터 초기화
                    kalman.statePost = (cv::Mat_<float>(4, 1) <<
                        ballCenter.x, ballCenter.y, 0, 0);
                    isKalmanInitialized = true;
                }

                // 측정값 업데이트
                measurement.at<float>(0) = ballCenter.x;
                measurement.at<float>(1) = ballCenter.y;

                // 칼만 필터로 보정
                cv::Mat corrected = kalman.correct(measurement);
                ballCenter.x = corrected.at<float>(0);
                ballCenter.y = corrected.at<float>(1);

                // 디버그 화면에 정보 표시
                if (!debugFrame.empty()) {
                    // 원본 컨투어 그리기
                    cv::drawContours(debugFrame, contours,
                        std::distance(contours.begin(), largestContour),
                        cv::Scalar(0, 255, 0), 2);

                    // 감지된 원 그리기
                    cv::circle(debugFrame, ballCenter, radius,
                        cv::Scalar(0, 255, 255), 2);

                    // 중심점 표시
                    cv::circle(debugFrame, ballCenter, 5,
                        cv::Scalar(0, 0, 255), -1);

                    // 정보 텍스트
                    std::string info = "Ball: (" + std::to_string(ballCenter.x) +
                        ", " + std::to_string(ballCenter.y) +
                        ") R:" + std::to_string((int)radius);
                    cv::putText(debugFrame, info, cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                }

                // 궤적 업데이트
                trajectory.push_back(ballCenter);
                if (trajectory.size() > maxTrajectorySize) {
                    trajectory.pop_front();
                }
            }
        }
        else if (isKalmanInitialized) {
            // 공을 못 찾았을 때는 예측값 사용
            prediction = kalman.predict();
            ballCenter.x = prediction.at<float>(0);
            ballCenter.y = prediction.at<float>(1);

            if (!debugFrame.empty()) {
                cv::circle(debugFrame, ballCenter, 20,
                    cv::Scalar(255, 0, 0), 2); // 파란색: 예측 위치
                cv::putText(debugFrame, "Predicted", ballCenter + cv::Point(10, -10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
            }
        }

        // 궤적 그리기
        if (!debugFrame.empty() && trajectory.size() > 1) {
            for (size_t i = 1; i < trajectory.size(); i++) {
                int thickness = (int)(i * 3.0 / trajectory.size()) + 1;
                cv::line(debugFrame, trajectory[i - 1], trajectory[i],
                    cv::Scalar(0, 165, 255), thickness);
            }
        }

        // 마스크 표시 (작은 창으로)
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

// HSV 색상 조절을 위한 트랙바 콜백
cv::Mat* g_frame = nullptr;
ColorBallTracker* g_tracker = nullptr;

void on_trackbar(int, void*) {
    // 트랙바 값으로 색상 범위 업데이트
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
    std::cout << "=== 색상 기반 공 추적 프로그램 ===" << std::endl;
    std::cout << "사용법:" << std::endl;
    std::cout << "- ESC: 종료" << std::endl;
    std::cout << "- SPACE: 궤적 초기화" << std::endl;
    std::cout << "- 트랙바로 색상 범위 조절" << std::endl;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "웹캠을 열 수 없습니다!" << std::endl;
        return -1;
    }

    // 해상도 설정
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS, 60);

    ColorBallTracker tracker;
    g_tracker = &tracker;

    cv::namedWindow("Ball Tracking", cv::WINDOW_NORMAL);
    cv::namedWindow("Controls", cv::WINDOW_NORMAL);
    cv::resizeWindow("Controls", 400, 300);

    // HSV 조절 트랙바
    cv::createTrackbar("H Low", "Controls", nullptr, 180, on_trackbar);
    cv::createTrackbar("H High", "Controls", nullptr, 180, on_trackbar);
    cv::createTrackbar("S Low", "Controls", nullptr, 255, on_trackbar);
    cv::createTrackbar("S High", "Controls", nullptr, 255, on_trackbar);
    cv::createTrackbar("V Low", "Controls", nullptr, 255, on_trackbar);
    cv::createTrackbar("V High", "Controls", nullptr, 255, on_trackbar);

    // 초기값 설정 (주황색)
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

        // 공 추적
        cv::Point ballPos = tracker.trackBall(frame, displayFrame);

        // FPS 계산
        frame_count++;
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - start_time).count();

        if (elapsed > 1000) {
            fps = frame_count * 1000.0 / elapsed;
            frame_count = 0;
            start_time = current_time;
        }

        // FPS 표시
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