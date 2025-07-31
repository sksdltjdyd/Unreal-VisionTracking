#include "opencv.h"

ColorBallTracker::ColorBallTracker() {
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

void ColorBallTracker::setColorRange(const cv::Scalar& lower, const cv::Scalar& upper) {
    lowerBound = lower;
    upperBound = upper;
}

void ColorBallTracker::clearTrajectory() {
    trajectory.clear();
}

