// opencv_tracker_demo.cpp
// OpenCV 내장 트래커 비교 프로그램

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <map>
#include <chrono>

class TrackerDemo {
private:
    cv::Ptr<cv::Tracker> tracker;
    cv::Rect2d bbox;
    bool isTracking = false;
    std::string trackerType;

    // 트래커별 성능 측정
    double avgFPS = 0.0;
    int frameCount = 0;

public:
    // 사용 가능한 트래커 목록
    std::map<std::string, std::string> trackerInfo = {
        {"CSRT", "정확도 높음, 속도 중간 (30 FPS)"},
        {"KCF", "속도 빠름, 정확도 중간 (100+ FPS)"},
        {"MOSSE", "매우 빠름, 정확도 낮음 (300+ FPS)"},
        {"MIL", "다중 인스턴스 학습"},
        {"BOOSTING", "오래된 방법, 느림"},
        {"MEDIANFLOW", "움직임 예측 좋음"},
        {"TLD", "장기 추적, 재탐지 가능"}
    };

    bool createTracker(const std::string& type) {
        trackerType = type;

        if (type == "CSRT") {
            tracker = cv::TrackerCSRT::create();
        }
        else if (type == "KCF") {
            tracker = cv::TrackerKCF::create();
        }
        else if (type == "MOSSE") {
            tracker = cv::TrackerMOSSE::create();
        }
        else if (type == "MIL") {
            tracker = cv::TrackerMIL::create();
        }
        else if (type == "BOOSTING") {
            tracker = cv::TrackerBoosting::create();
        }
        else if (type == "MEDIANFLOW") {
            tracker = cv::TrackerMedianFlow::create();
        }
        else if (type == "TLD") {
            tracker = cv::TrackerTLD::create();
        }
        else {
            std::cerr << "Unknown tracker type: " << type << std::endl;
            return false;
        }

        isTracking = false;
        frameCount = 0;
        avgFPS = 0.0;
        return true;
    }

    void initTracking(const cv::Mat& frame, const cv::Rect2d& initialBbox) {
        bbox = initialBbox;
        tracker->init(frame, bbox);
        isTracking = true;
    }

    bool update(const cv::Mat& frame) {
        if (!isTracking) return false;

        auto start = std::chrono::high_resolution_clock::now();
        bool success = tracker->update(frame, bbox);
        auto end = std::chrono::high_resolution_clock::now();

        // FPS 계산
        double fps = 1000.0 / std::chrono::duration<double, std::milli>(end - start).count();
        avgFPS = (avgFPS * frameCount + fps) / (frameCount + 1);
        frameCount++;

        return success;
    }

    void drawResult(cv::Mat& frame, bool success) {
        if (success && isTracking) {
            // 추적 성공 - 초록색 박스
            cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);

            // 중심점 표시
            cv::Point center(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
            cv::circle(frame, center, 4, cv::Scalar(0, 255, 0), -1);

            // 트래커 정보 표시
            std::string info = trackerType + " | FPS: " + std::to_string((int)avgFPS);
            cv::putText(frame, info, cv::Point(bbox.x, bbox.y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        }
        else if (isTracking) {
            // 추적 실패 - 빨간색 텍스트
            cv::putText(frame, "Tracking Failed!", cv::Point(100, 100),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }
    }

    cv::Rect2d getBbox() const { return bbox; }
    bool getIsTracking() const { return isTracking; }
    double getAvgFPS() const { return avgFPS; }
};

// 마우스로 추적 영역 선택
bool selectObject = false;
cv::Rect2d selectedBbox;
cv::Point startPoint;

void onMouse(int event, int x, int y, int, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        selectObject = true;
        startPoint = cv::Point(x, y);
        selectedBbox = cv::Rect2d(x, y, 0, 0);
    }
    else if (selectObject && event == cv::EVENT_MOUSEMOVE) {
        selectedBbox.x = std::min(x, startPoint.x);
        selectedBbox.y = std::min(y, startPoint.y);
        selectedBbox.width = std::abs(x - startPoint.x);
        selectedBbox.height = std::abs(y - startPoint.y);
    }
    else if (selectObject && event == cv::EVENT_LBUTTONUP) {
        selectObject = false;
        if (selectedBbox.width > 0 && selectedBbox.height > 0) {
            // 선택 완료
        }
    }
}

int main() {
    std::cout << "=== OpenCV 트래커 비교 프로그램 ===" << std::endl;
    std::cout << "\n사용법:" << std::endl;
    std::cout << "- 마우스로 추적할 영역 선택" << std::endl;
    std::cout << "- 숫자 키로 트래커 변경:" << std::endl;
    std::cout << "  1: CSRT (추천 - 정확도 높음)" << std::endl;
    std::cout << "  2: KCF (빠름)" << std::endl;
    std::cout << "  3: MOSSE (매우 빠름)" << std::endl;
    std::cout << "  4: MIL" << std::endl;
    std::cout << "  5: BOOSTING" << std::endl;
    std::cout << "  6: MEDIANFLOW" << std::endl;
    std::cout << "  7: TLD" << std::endl;
    std::cout << "- SPACE: 추적 초기화" << std::endl;
    std::cout << "- ESC: 종료\n" << std::endl;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "웹캠을 열 수 없습니다!" << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    cv::namedWindow("Tracker Demo", cv::WINDOW_NORMAL);
    cv::setMouseCallback("Tracker Demo", onMouse);

    TrackerDemo trackerDemo;
    trackerDemo.createTracker("CSRT"); // 기본 트래커

    cv::Mat frame;
    bool paused = false;

    while (true) {
        if (!paused) {
            cap >> frame;
            if (frame.empty()) continue;
        }

        cv::Mat display = frame.clone();

        // 선택 중인 박스 표시
        if (selectObject && selectedBbox.width > 0 && selectedBbox.height > 0) {
            cv::rectangle(display, selectedBbox, cv::Scalar(255, 0, 0), 2);
            paused = true; // 선택 중에는 일시정지
        }

        // 트래킹 업데이트
        if (trackerDemo.getIsTracking() && !selectObject) {
            bool success = trackerDemo.update(frame);
            trackerDemo.drawResult(display, success);
            paused = false;
        }

        // 안내 텍스트
        if (!trackerDemo.getIsTracking()) {
            cv::putText(display, "Draw a box around the object to track",
                cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX,
                0.8, cv::Scalar(255, 255, 0), 2);
        }

        // 현재 트래커 정보
        cv::putText(display, "Current: " + trackerDemo.trackerType,
            cv::Point(20, display.rows - 20), cv::FONT_HERSHEY_SIMPLEX,
            0.7, cv::Scalar(200, 200, 200), 2);

        cv::imshow("Tracker Demo", display);

        char key = cv::waitKey(1);

        // 키보드 입력 처리
        if (key == 27) break; // ESC
        else if (key == ' ') { // SPACE - 초기화
            trackerDemo.createTracker(trackerDemo.trackerType);
            selectedBbox = cv::Rect2d();
        }
        else if (key >= '1' && key <= '7') { // 트래커 변경
            std::vector<std::string> trackerTypes = {
                "CSRT", "KCF", "MOSSE", "MIL", "BOOSTING", "MEDIANFLOW", "TLD"
            };
            int index = key - '1';
            if (index < trackerTypes.size()) {
                trackerDemo.createTracker(trackerTypes[index]);
                std::cout << "트래커 변경: " << trackerTypes[index] << std::endl;
            }
        }

        // 선택 완료 시 트래킹 시작
        if (!selectObject && selectedBbox.width > 0 && selectedBbox.height > 0
            && !trackerDemo.getIsTracking()) {
            trackerDemo.initTracking(frame, selectedBbox);
            selectedBbox = cv::Rect2d(); // 리셋
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}