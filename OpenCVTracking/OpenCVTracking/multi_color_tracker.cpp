// multi_color_tracker_with_trackbar.cpp

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <deque>
#include <map>

// 개별 물체 추적을 위한 구조체
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

// 색상 범위 정의 구조체
struct ColorRange {
    std::string name;
    cv::Scalar lowerBound;
    cv::Scalar upperBound;
    cv::Scalar displayColor;

    // HSV 값 저장 (트랙바용)
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

// 전역 변수 (트랙바 콜백용)
class MultiColorTracker* g_tracker = nullptr;
int g_selectedColor = 0;

// 트랙바 콜백 함수
void on_trackbar(int, void*) {
    if (g_tracker != nullptr) {
        // MultiColorTracker의 updateCurrentColorRange 호출
        // (아래에서 정의)
    }
}

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
        colorRanges.push_back(ColorRange("Orange", 10, 20, 100, 255, 100, 255,
            cv::Scalar(0, 165, 255)));
    }

    void createTrackbars() {
        cv::namedWindow("HSV Controls", cv::WINDOW_NORMAL);
        cv::resizeWindow("HSV Controls", 400, 400);

        // 색상 선택 트랙바
        cv::createTrackbar("Color Select", "HSV Controls", &g_selectedColor,
            colorRanges.size() - 1, on_trackbar);

        updateTrackbars();
    }

    void updateTrackbars() {
        if (currentColorIndex >= colorRanges.size()) return;

        ColorRange& cr = colorRanges[currentColorIndex];

        // 기존 트랙바 삭제하고 새로 생성
        cv::destroyWindow("HSV Controls");
        cv::namedWindow("HSV Controls", cv::WINDOW_NORMAL);
        cv::resizeWindow("HSV Controls", 400, 400);

        // 색상 선택
        cv::createTrackbar("Color Select", "HSV Controls", &g_selectedColor,
            colorRanges.size() - 1, on_trackbar);

        // HSV 트랙바
        cv::createTrackbar("H Min", "HSV Controls", &cr.h_min, 180, on_trackbar);
        cv::createTrackbar("H Max", "HSV Controls", &cr.h_max, 180, on_trackbar);
        cv::createTrackbar("S Min", "HSV Controls", &cr.s_min, 255, on_trackbar);
        cv::createTrackbar("S Max", "HSV Controls", &cr.s_max, 255, on_trackbar);
        cv::createTrackbar("V Min", "HSV Controls", &cr.v_min, 255, on_trackbar);
        cv::createTrackbar("V Max", "HSV Controls", &cr.v_max, 255, on_trackbar);
    }

    void updateCurrentColorRange() {
        // 선택된 색상이 변경되었는지 확인
        if (g_selectedColor != currentColorIndex) {
            currentColorIndex = g_selectedColor;
            updateTrackbars();
        }
        else {
            // 현재 색상의 범위 업데이트
            if (currentColorIndex < colorRanges.size()) {
                colorRanges[currentColorIndex].updateBounds();
            }
        }
    }

    void trackObjects(const cv::Mat& frame, cv::Mat& debugFrame) {
        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        // 트랙바 값 반영
        updateCurrentColorRange();

        // 각 색상별로 추적된 물체들을 임시 저장
        std::map<std::string, std::vector<cv::Point2f>> detectedObjects;

        // 모든 색상 범위에 대해 검사
        for (size_t colorIdx = 0; colorIdx < colorRanges.size(); colorIdx++) {
            ColorRange& colorRange = colorRanges[colorIdx];
            cv::Mat mask;

            // 빨간색은 특별 처리 (0도와 180도 근처)
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

            // 현재 선택된 색상의 마스크 표시
            if (colorIdx == currentColorIndex) {
                cv::Mat smallMask;
                cv::resize(mask, smallMask, cv::Size(320, 240));
                cv::imshow("Current Color Mask", smallMask);
            }

            // 노이즈 제거
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
            cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
            cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

            // 컨투어 찾기
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            // 유효한 컨투어 처리
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

        // 기존 추적 물체 업데이트
        for (std::map<std::string, TrackedObject>::iterator it = trackedObjects.begin();
            it != trackedObjects.end(); ++it) {
            TrackedObject& obj = it->second;
            bool matched = false;

            std::map<std::string, std::vector<cv::Point2f>>::iterator detIt =
                detectedObjects.find(obj.colorName);

            if (detIt != detectedObjects.end()) {
                std::vector<cv::Point2f>& positions = detIt->second;
                if (!positions.empty()) {
                    float minDist = FLT_MAX;
                    cv::Point2f closestPos;
                    int closestIdx = -1;

                    cv::Mat prediction = obj.kalman.predict();
                    cv::Point2f predictedPos(prediction.at<float>(0), prediction.at<float>(1));

                    for (int i = 0; i < positions.size(); i++) {
                        float dist = cv::norm(positions[i] - predictedPos);
                        if (dist < minDist && dist < 100) {
                            minDist = dist;
                            closestPos = positions[i];
                            closestIdx = i;
                        }
                    }

                    if (closestIdx >= 0) {
                        cv::Mat measurement = (cv::Mat_<float>(2, 1) <<
                            closestPos.x, closestPos.y);
                        cv::Mat corrected = obj.kalman.correct(measurement);
                        obj.position = cv::Point2f(corrected.at<float>(0),
                            corrected.at<float>(1));

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
                    obj.trajectory.clear();
                }
                else {
                    cv::Mat prediction = obj.kalman.predict();
                    obj.position = cv::Point2f(prediction.at<float>(0),
                        prediction.at<float>(1));
                }
            }
        }

        // 새로운 물체 추가
        for (std::map<std::string, std::vector<cv::Point2f>>::const_iterator it =
            detectedObjects.begin(); it != detectedObjects.end(); ++it) {
            const std::string& colorName = it->first;
            const std::vector<cv::Point2f>& positions = it->second;

            for (size_t i = 0; i < positions.size(); i++) {
                const cv::Point2f& pos = positions[i];
                std::string objKey = colorName + "_" + std::to_string(nextObjectId);

                cv::Scalar displayColor(255, 255, 255);
                for (const auto& cr : colorRanges) {
                    if (cr.name == colorName) {
                        displayColor = cr.displayColor;
                        break;
                    }
                }

                trackedObjects[objKey] = TrackedObject(nextObjectId++, colorName, displayColor);
                TrackedObject& newObj = trackedObjects[objKey];

                newObj.kalman.statePost = (cv::Mat_<float>(4, 1) <<
                    pos.x, pos.y, 0, 0);
                newObj.position = pos;
                newObj.isActive = true;
                newObj.trajectory.push_back(cv::Point(pos));
            }
        }

        // 비활성 물체 제거
        for (std::map<std::string, TrackedObject>::iterator it = trackedObjects.begin();
            it != trackedObjects.end(); ) {
            if (!it->second.isActive && it->second.lostFrames > maxLostFrames * 2) {
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

    void drawColorInfo(cv::Mat& frame) {
        // 현재 선택된 색상 정보 표시
        if (currentColorIndex < colorRanges.size()) {
            ColorRange& cr = colorRanges[currentColorIndex];
            std::string info = "Adjusting: " + cr.name +
                " [H:" + std::to_string(cr.h_min) + "-" + std::to_string(cr.h_max) +
                " S:" + std::to_string(cr.s_min) + "-" + std::to_string(cr.s_max) +
                " V:" + std::to_string(cr.v_min) + "-" + std::to_string(cr.v_max) + "]";

            cv::rectangle(frame, cv::Point(10, frame.rows - 40),
                cv::Point(600, frame.rows - 10),
                cv::Scalar(0, 0, 0), -1);
            cv::putText(frame, info, cv::Point(15, frame.rows - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cr.displayColor, 2);
        }
    }

    void drawTrackingInfo(cv::Mat& frame) {
        int activeCount = 0;

        for (std::map<std::string, TrackedObject>::const_iterator it = trackedObjects.begin();
            it != trackedObjects.end(); ++it) {
            const TrackedObject& obj = it->second;

            if (obj.isActive) {
                activeCount++;

                cv::circle(frame, cv::Point(obj.position), 8, obj.displayColor, -1);
                cv::circle(frame, cv::Point(obj.position), 10, cv::Scalar(255, 255, 255), 2);

                std::string label = obj.colorName + " #" + std::to_string(obj.id);
                cv::putText(frame, label,
                    cv::Point(obj.position) + cv::Point(15, -10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, obj.displayColor, 2);

                std::string posText = "(" + std::to_string(int(obj.position.x)) +
                    "," + std::to_string(int(obj.position.y)) + ")";
                cv::putText(frame, posText,
                    cv::Point(obj.position) + cv::Point(15, 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);

                if (obj.trajectory.size() > 1) {
                    for (size_t i = 1; i < obj.trajectory.size(); i++) {
                        int thickness = (int)(i * 3.0 / obj.trajectory.size()) + 1;
                        cv::line(frame, obj.trajectory[i - 1], obj.trajectory[i],
                            obj.displayColor, thickness);
                    }
                }
            }
            else if (obj.lostFrames <= maxLostFrames) {
                cv::circle(frame, cv::Point(obj.position), 15, obj.displayColor, 1);
                cv::putText(frame, "Lost #" + std::to_string(obj.id),
                    cv::Point(obj.position) + cv::Point(15, -10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, obj.displayColor, 1);
            }
        }

        std::string statusText = "Active Objects: " + std::to_string(activeCount) +
            " / Total: " + std::to_string(trackedObjects.size());
        cv::putText(frame, statusText, cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    }

    void clearAllTracks() {
        trackedObjects.clear();
        nextObjectId = 0;
    }

    void saveColorSettings() {
        std::cout << "\n=== 색상 설정 저장 ===" << std::endl;
        for (const auto& cr : colorRanges) {
            std::cout << cr.name << ": " << std::endl;
            std::cout << "  H: " << cr.h_min << "-" << cr.h_max << std::endl;
            std::cout << "  S: " << cr.s_min << "-" << cr.s_max << std::endl;
            std::cout << "  V: " << cr.v_min << "-" << cr.v_max << std::endl;
        }
        std::cout << "=====================\n" << std::endl;
    }
};


int main() {
    std::cout << "=== HSV 조절 가능한 멀티 컬러 추적 ===" << std::endl;
    std::cout << "사용법:" << std::endl;
    std::cout << "- Color Select 트랙바로 조절할 색상 선택" << std::endl;
    std::cout << "- 각 HSV 트랙바로 색상 범위 조절" << std::endl;
    std::cout << "- 's': 현재 색상 설정 저장" << std::endl;
    std::cout << "- SPACE: 모든 추적 초기화" << std::endl;
    std::cout << "- ESC: 종료" << std::endl << std::endl;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "웹캠을 열 수 없습니다!" << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS, 60);

    MultiColorTracker tracker;
    tracker.createTrackbars();

    cv::namedWindow("Multi Color Tracking", cv::WINDOW_NORMAL);
    cv::namedWindow("Current Color Mask", cv::WINDOW_NORMAL);
    cv::resizeWindow("Current Color Mask", 320, 240);

    cv::Mat frame;
    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    double fps = 0.0;

    while (true) {
        cap >> frame;
        if (frame.empty()) continue;

        cv::Mat displayFrame = frame.clone();

        tracker.trackObjects(frame, displayFrame);

        frame_count++;
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - start_time).count();

        if (elapsed > 1000) {
            fps = frame_count * 1000.0 / elapsed;
            frame_count = 0;
            start_time = current_time;
        }

        cv::putText(displayFrame, "FPS: " + std::to_string((int)fps),
            cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX,
            0.7, cv::Scalar(255, 255, 0), 2);

        cv::imshow("Multi Color Tracking", displayFrame);

        char key = cv::waitKey(1);
        if (key == 27) break;  // ESC
        else if (key == ' ') tracker.clearAllTracks();  // SPACE
        else if (key == 's' || key == 'S') tracker.saveColorSettings();  // Save
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}