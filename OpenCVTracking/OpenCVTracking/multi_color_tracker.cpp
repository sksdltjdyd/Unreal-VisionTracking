// multi_color_tracker_with_trackbar.cpp
// HSV 조절바가 포함된 멀티 컬러 트래커 (동시 추적 시각화 개선 버전)

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <deque>
#include <map>
#include <chrono> // FPS 계산을 위해 추가

// ... (TrackedObject, ColorRange 구조체는 변경 없음) ...
struct TrackedObject {
    int id;
    std::string colorName;
    cv::Scalar displayColor;
    cv::Point2f position;
    cv::KalmanFilter kalman;
    std::deque<cv::Point> trajectory;
    bool isActive;
    int lostFrames;

    TrackedObject(int objId, const std::string& color, const cv::Scalar& dispColor)
        : id(objId), colorName(color), displayColor(dispColor),
        isActive(false), lostFrames(0) {

        kalman.init(4, 2, 0);
        kalman.transitionMatrix = (cv::Mat_<float>(4, 4) <<
            1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1);
        kalman.measurementMatrix = cv::Mat::eye(2, 4, CV_32F);
        cv::setIdentity(kalman.processNoiseCov, cv::Scalar::all(0.1));
        cv::setIdentity(kalman.measurementNoiseCov, cv::Scalar::all(10));
        cv::setIdentity(kalman.errorCovPost, cv::Scalar::all(1));
    }

    TrackedObject() : id(0), isActive(false), lostFrames(0) {}
};

struct ColorRange {
    std::string name;
    cv::Scalar lowerBound;
    cv::Scalar upperBound;
    cv::Scalar displayColor;

    int h_min, h_max;
    int s_min, s_max;
    int v_min, v_max;

    ColorRange(const std::string& n, int hmin, int hmax, int smin, int smax,
        int vmin, int vmax, const cv::Scalar& color)
        : name(n), displayColor(color),
        h_min(hmin), h_max(hmax),
        s_min(smin), s_max(smax),
        v_min(vmin), v_max(vmax) {
        updateBounds();
    }

    void updateBounds() {
        lowerBound = cv::Scalar(h_min, s_min, v_min);
        upperBound = cv::Scalar(h_max, s_max, v_max);
    }
};

class MultiColorTracker; // 전방 선언
MultiColorTracker* g_tracker = nullptr;
int g_selectedColor = 0;

void on_trackbar(int, void*); // 전방 선언

class MultiColorTracker {
private:
    std::vector<ColorRange> colorRanges;
    std::map<std::string, TrackedObject> trackedObjects;

    const int maxTrajectorySize = 50;
    const double minContourArea = 500.0;
    const int maxLostFrames = 10;

    int nextObjectId = 0;
    int currentColorIndex = 0;

public:
    MultiColorTracker() {
        g_tracker = this;
        initializeDefaultColors();
    }

    void initializeDefaultColors() {
        // 색상 추가 (이름, H최소, H최대, S최소, S최대, V최소, V최대, 표시색상)
        colorRanges.push_back(ColorRange("Red", 0, 10, 100, 255, 100, 255,
            cv::Scalar(0, 0, 255)));
        colorRanges.push_back(ColorRange("Blue", 100, 120, 100, 255, 100, 255,
            cv::Scalar(255, 0, 0)));
        colorRanges.push_back(ColorRange("Green", 40, 80, 100, 255, 100, 255,
            cv::Scalar(0, 255, 0)));
        colorRanges.push_back(ColorRange("Yellow", 20, 40, 100, 255, 100, 255,
            cv::Scalar(0, 255, 255)));
    }

    void createTrackbars() {
        cv::namedWindow("HSV Controls", cv::WINDOW_NORMAL);
        cv::resizeWindow("HSV Controls", 400, 400);

        // 색상 선택 트랙바를 처음에 한 번만 생성
        cv::createTrackbar("Color Select", "HSV Controls", &g_selectedColor,
            colorRanges.size() - 1, on_trackbar);

        // 초기 색상에 대한 HSV 트랙바 생성
        updateHsvTrackbars();
    }

    void updateHsvTrackbars() {
        if (currentColorIndex >= colorRanges.size()) return;
        ColorRange& cr = colorRanges[currentColorIndex];

        // HSV 트랙바 생성 (창을 매번 파괴하고 다시 만들 필요 없음)
        cv::createTrackbar("H Min", "HSV Controls", &cr.h_min, 180, on_trackbar);
        cv::createTrackbar("H Max", "HSV Controls", &cr.h_max, 180, on_trackbar);
        cv::createTrackbar("S Min", "HSV Controls", &cr.s_min, 255, on_trackbar);
        cv::createTrackbar("S Max", "HSV Controls", &cr.s_max, 255, on_trackbar);
        cv::createTrackbar("V Min", "HSV Controls", &cr.v_min, 255, on_trackbar);
        cv::createTrackbar("V Max", "HSV Controls", &cr.v_max, 255, on_trackbar);
    }

    void updateCurrentColorRange() {
        if (g_selectedColor != currentColorIndex) {
            currentColorIndex = g_selectedColor;
            updateHsvTrackbars(); // 선택된 색상에 맞게 HSV 트랙바 업데이트
        }

        // 현재 선택된 색상의 범위 값 업데이트
        if (currentColorIndex < colorRanges.size()) {
            colorRanges[currentColorIndex].updateBounds();
        }
    }

    void trackObjects(const cv::Mat& frame, cv::Mat& debugFrame) {
        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        updateCurrentColorRange();

        // [추가] 모든 마스크를 합쳐서 시각화하기 위한 Mat
        cv::Mat totalMask = cv::Mat::zeros(hsv.size(), CV_8UC1);

        std::map<std::string, std::vector<cv::Point2f>> detectedObjects;

        for (size_t colorIdx = 0; colorIdx < colorRanges.size(); colorIdx++) {
            ColorRange& colorRange = colorRanges[colorIdx];
            cv::Mat mask;

            if (colorRange.name == "Red" && colorRange.h_min <= 10) {
                cv::Mat mask1, mask2;
                cv::inRange(hsv, cv::Scalar(0, colorRange.s_min, colorRange.v_min),
                    cv::Scalar(colorRange.h_max, colorRange.s_max, colorRange.v_max), mask1);
                cv::inRange(hsv, cv::Scalar(170, colorRange.s_min, colorRange.v_min),
                    cv::Scalar(180, colorRange.s_max, colorRange.v_max), mask2);
                mask = mask1 | mask2;
            }
            else {
                cv::inRange(hsv, colorRange.lowerBound, colorRange.upperBound, mask);
            }

            if (colorIdx == currentColorIndex) {
                cv::imshow("Current Color Mask", mask);
            }

            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
            cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 2);
            cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);

            // [추가] 현재 마스크를 전체 마스크에 합침
            totalMask |= mask;

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            for (const auto& contour : contours) {
                double area = cv::contourArea(contour);
                if (area > minContourArea) {
                    cv::Moments m = cv::moments(contour);
                    if (m.m00 != 0) {
                        cv::Point2f center(m.m10 / m.m00, m.m01 / m.m00);
                        detectedObjects[colorRange.name].push_back(center);

                        if (!debugFrame.empty()) {
                            cv::drawContours(debugFrame, std::vector<std::vector<cv::Point>>{contour},
                                -1, colorRange.displayColor, 2);
                        }
                    }
                }
            }
        }

        // [추가] 합쳐진 마스크를 새 창에 표시
        cv::imshow("All Masks Combined", totalMask);

        // ... (물체 매칭, 추가, 제거 로직은 변경 없음) ...
        // (기존 추적 물체 업데이트 로직)
        for (auto& pair : trackedObjects) {
            TrackedObject& obj = pair.second;
            bool matched = false;

            auto detIt = detectedObjects.find(obj.colorName);
            if (detIt != detectedObjects.end()) {
                auto& positions = detIt->second;
                if (!positions.empty()) {
                    float minDist = FLT_MAX;
                    int closestIdx = -1;

                    cv::Mat prediction = obj.kalman.predict();
                    cv::Point2f predictedPos(prediction.at<float>(0), prediction.at<float>(1));

                    for (int i = 0; i < positions.size(); i++) {
                        float dist = cv::norm(positions[i] - predictedPos);
                        if (dist < minDist && dist < 100) { // 임계값
                            minDist = dist;
                            closestIdx = i;
                        }
                    }

                    if (closestIdx != -1) {
                        cv::Point2f closestPos = positions[closestIdx];
                        cv::Mat measurement = (cv::Mat_<float>(2, 1) << closestPos.x, closestPos.y);
                        cv::Mat corrected = obj.kalman.correct(measurement);
                        obj.position = cv::Point2f(corrected.at<float>(0), corrected.at<float>(1));

                        obj.trajectory.push_back(cv::Point(obj.position));
                        if (obj.trajectory.size() > maxTrajectorySize) {
                            obj.trajectory.pop_front();
                        }

                        obj.lostFrames = 0;
                        obj.isActive = true;
                        matched = true;

                        positions.erase(positions.begin() + closestIdx);
                    }
                }
            }

            if (!matched) {
                obj.lostFrames++;
                if (obj.lostFrames > maxLostFrames) {
                    obj.isActive = false;
                }
                else {
                    cv::Mat prediction = obj.kalman.predict();
                    obj.position = cv::Point2f(prediction.at<float>(0), prediction.at<float>(1));
                    obj.trajectory.push_back(cv::Point(obj.position)); // 예측된 경로도 추가
                    if (obj.trajectory.size() > maxTrajectorySize) {
                        obj.trajectory.pop_front();
                    }
                }
            }
        }

        // (새로운 물체 추가 로직)
        for (const auto& pair : detectedObjects) {
            const std::string& colorName = pair.first;
            const auto& positions = pair.second;

            cv::Scalar displayColor;
            for (const auto& cr : colorRanges) {
                if (cr.name == colorName) {
                    displayColor = cr.displayColor;
                    break;
                }
            }

            for (const auto& pos : positions) {
                std::string objKey = colorName + "_" + std::to_string(nextObjectId);
                trackedObjects[objKey] = TrackedObject(nextObjectId++, colorName, displayColor);
                TrackedObject& newObj = trackedObjects[objKey];

                newObj.kalman.statePost = (cv::Mat_<float>(4, 1) << pos.x, pos.y, 0, 0);
                newObj.position = pos;
                newObj.isActive = true;
                newObj.trajectory.push_back(cv::Point(pos));
            }
        }

        // (비활성 물체 제거 로직)
        for (auto it = trackedObjects.begin(); it != trackedObjects.end(); ) {
            if (!it->second.isActive && it->second.lostFrames > maxLostFrames) {
                it = trackedObjects.erase(it);
            }
            else {
                ++it;
            }
        }

        if (!debugFrame.empty()) {
            drawTrackingInfo(debugFrame);
            drawColorInfo(debugFrame);
        }
    }

    // ... (drawColorInfo, drawTrackingInfo, clearAllTracks, saveColorSettings 함수는 변경 없음) ...
    void drawColorInfo(cv::Mat& frame) {
        if (currentColorIndex < colorRanges.size()) {
            ColorRange& cr = colorRanges[currentColorIndex];
            std::string info = "Adjusting: " + cr.name +
                " [H:" + std::to_string(cr.h_min) + "-" + std::to_string(cr.h_max) +
                " S:" + std::to_string(cr.s_min) + "-" + std::to_string(cr.s_max) +
                " V:" + std::to_string(cr.v_min) + "-" + std::to_string(cr.v_max) + "]";

            cv::rectangle(frame, cv::Point(10, frame.rows - 40),
                cv::Point(frame.cols - 10, frame.rows - 10),
                cv::Scalar(0, 0, 0), -1);
            cv::putText(frame, info, cv::Point(15, frame.rows - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cr.displayColor, 2);
        }
    }

    void drawTrackingInfo(cv::Mat& frame) {
        int activeCount = 0;
        for (const auto& pair : trackedObjects) {
            const TrackedObject& obj = pair.second;
            if (obj.isActive) {
                activeCount++;
                cv::circle(frame, cv::Point(obj.position), 8, obj.displayColor, -1);
                cv::circle(frame, cv::Point(obj.position), 10, cv::Scalar(255, 255, 255), 2);

                std::string label = obj.colorName + " #" + std::to_string(obj.id);
                cv::putText(frame, label,
                    cv::Point(obj.position) + cv::Point(15, -10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, obj.displayColor, 2);

                if (obj.trajectory.size() > 1) {
                    for (size_t i = 1; i < obj.trajectory.size(); i++) {
                        float a = (float)i / obj.trajectory.size();
                        cv::line(frame, obj.trajectory[i - 1], obj.trajectory[i],
                            obj.displayColor, 2, cv::LINE_AA, 0);
                    }
                }
            }
            else if (obj.lostFrames <= maxLostFrames) {
                cv::circle(frame, cv::Point(obj.position), 15, obj.displayColor, 1);
                cv::putText(frame, "?",
                    cv::Point(obj.position) - cv::Point(5, 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, obj.displayColor, 2);
            }
        }
        std::string statusText = "Active: " + std::to_string(activeCount) +
            " / Total: " + std::to_string(trackedObjects.size());
        cv::putText(frame, statusText, cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    }

    void clearAllTracks() {
        trackedObjects.clear();
        nextObjectId = 0;
    }

    void saveColorSettings() {
        std::cout << "\n=== Color Settings Saved ===\n";
        for (const auto& cr : colorRanges) {
            std::cout << cr.name << ": H(" << cr.h_min << "," << cr.h_max
                << "), S(" << cr.s_min << "," << cr.s_max
                << "), V(" << cr.v_min << "," << cr.v_max << ")\n";
        }
        std::cout << "==========================\n\n";
    }
};

// 트랙바 콜백 구현
void on_trackbar(int, void*) {
    if (g_tracker != nullptr) {
        g_tracker->updateCurrentColorRange();
    }
}

int main() {
    std::cout << "=== Multi Color Tracker with HSV Controls ===" << std::endl;
    // ... (사용법 안내 문구) ...

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Could not open webcam!" << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    MultiColorTracker tracker;
    tracker.createTrackbars();

    cv::namedWindow("Multi Color Tracking", cv::WINDOW_NORMAL);
    cv::namedWindow("Current Color Mask", cv::WINDOW_NORMAL);

    // [추가] 모든 마스크를 표시할 창 생성
    cv::namedWindow("All Masks Combined", cv::WINDOW_NORMAL);

    cv::Mat frame;
    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    double fps = 0.0;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::flip(frame, frame, 1); // 좌우 반전
        cv::Mat displayFrame = frame.clone();

        tracker.trackObjects(frame, displayFrame);

        // FPS 계산 및 표시
        frame_count++;
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();
        if (elapsed > 1000) {
            fps = frame_count * 1000.0 / elapsed;
            frame_count = 0;
            start_time = current_time;
        }
        cv::putText(displayFrame, "FPS: " + std::to_string((int)fps),
            cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX,
            0.7, cv::Scalar(0, 255, 255), 2);

        cv::imshow("Multi Color Tracking", displayFrame);

        char key = cv::waitKey(1);
        if (key == 27) break; // ESC
        else if (key == ' ') tracker.clearAllTracks(); // SPACE
        else if (key == 's' || key == 'S') tracker.saveColorSettings(); // 's'
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}