#ifndef __OPENCV_H__
#define __OPENCV_H__

#include <opencv2/opencv.hpp>

class ColorBallTracker {
private:
    

public:
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

    ColorBallTracker();

    void setColorRange(const cv::Scalar& lower, const cv::Scalar& upper);

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

    void clearTrajectory();

};

#endif // !__OPENCV_H__
