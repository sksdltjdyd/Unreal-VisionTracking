#include "opencv.h"

ColorBallTracker::ColorBallTracker() {
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

void ColorBallTracker::setColorRange(const cv::Scalar& lower, const cv::Scalar& upper) {
    lowerBound = lower;
    upperBound = upper;
}

void ColorBallTracker::clearTrajectory() {
    trajectory.clear();
}

