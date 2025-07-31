#ifndef __OPENCV_H__
#define __OPENCV_H__

#include <opencv2/opencv.hpp>

class ColorBallTracker {
private:
    

public:
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

    ColorBallTracker();

    void setColorRange(const cv::Scalar& lower, const cv::Scalar& upper);

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

    void clearTrajectory();

};

#endif // !__OPENCV_H__
