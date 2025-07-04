// opencv_tracker_demo.cpp
// OpenCV ���� Ʈ��Ŀ �� ���α׷�

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

    // Ʈ��Ŀ�� ���� ����
    double avgFPS = 0.0;
    int frameCount = 0;

public:
    // ��� ������ Ʈ��Ŀ ���
    std::map<std::string, std::string> trackerInfo = {
        {"CSRT", "��Ȯ�� ����, �ӵ� �߰� (30 FPS)"},
        {"KCF", "�ӵ� ����, ��Ȯ�� �߰� (100+ FPS)"},
        {"MOSSE", "�ſ� ����, ��Ȯ�� ���� (300+ FPS)"},
        {"MIL", "���� �ν��Ͻ� �н�"},
        {"BOOSTING", "������ ���, ����"},
        {"MEDIANFLOW", "������ ���� ����"},
        {"TLD", "��� ����, ��Ž�� ����"}
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

        // FPS ���
        double fps = 1000.0 / std::chrono::duration<double, std::milli>(end - start).count();
        avgFPS = (avgFPS * frameCount + fps) / (frameCount + 1);
        frameCount++;

        return success;
    }

    void drawResult(cv::Mat& frame, bool success) {
        if (success && isTracking) {
            // ���� ���� - �ʷϻ� �ڽ�
            cv::rectangle(frame, bbox, cv::Scalar(0, 255, 0), 2);

            // �߽��� ǥ��
            cv::Point center(bbox.x + bbox.width / 2, bbox.y + bbox.height / 2);
            cv::circle(frame, center, 4, cv::Scalar(0, 255, 0), -1);

            // Ʈ��Ŀ ���� ǥ��
            std::string info = trackerType + " | FPS: " + std::to_string((int)avgFPS);
            cv::putText(frame, info, cv::Point(bbox.x, bbox.y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        }
        else if (isTracking) {
            // ���� ���� - ������ �ؽ�Ʈ
            cv::putText(frame, "Tracking Failed!", cv::Point(100, 100),
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }
    }

    cv::Rect2d getBbox() const { return bbox; }
    bool getIsTracking() const { return isTracking; }
    double getAvgFPS() const { return avgFPS; }
};

// ���콺�� ���� ���� ����
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
            // ���� �Ϸ�
        }
    }
}

int main() {
    std::cout << "=== OpenCV Ʈ��Ŀ �� ���α׷� ===" << std::endl;
    std::cout << "\n����:" << std::endl;
    std::cout << "- ���콺�� ������ ���� ����" << std::endl;
    std::cout << "- ���� Ű�� Ʈ��Ŀ ����:" << std::endl;
    std::cout << "  1: CSRT (��õ - ��Ȯ�� ����)" << std::endl;
    std::cout << "  2: KCF (����)" << std::endl;
    std::cout << "  3: MOSSE (�ſ� ����)" << std::endl;
    std::cout << "  4: MIL" << std::endl;
    std::cout << "  5: BOOSTING" << std::endl;
    std::cout << "  6: MEDIANFLOW" << std::endl;
    std::cout << "  7: TLD" << std::endl;
    std::cout << "- SPACE: ���� �ʱ�ȭ" << std::endl;
    std::cout << "- ESC: ����\n" << std::endl;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "��ķ�� �� �� �����ϴ�!" << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    cv::namedWindow("Tracker Demo", cv::WINDOW_NORMAL);
    cv::setMouseCallback("Tracker Demo", onMouse);

    TrackerDemo trackerDemo;
    trackerDemo.createTracker("CSRT"); // �⺻ Ʈ��Ŀ

    cv::Mat frame;
    bool paused = false;

    while (true) {
        if (!paused) {
            cap >> frame;
            if (frame.empty()) continue;
        }

        cv::Mat display = frame.clone();

        // ���� ���� �ڽ� ǥ��
        if (selectObject && selectedBbox.width > 0 && selectedBbox.height > 0) {
            cv::rectangle(display, selectedBbox, cv::Scalar(255, 0, 0), 2);
            paused = true; // ���� �߿��� �Ͻ�����
        }

        // Ʈ��ŷ ������Ʈ
        if (trackerDemo.getIsTracking() && !selectObject) {
            bool success = trackerDemo.update(frame);
            trackerDemo.drawResult(display, success);
            paused = false;
        }

        // �ȳ� �ؽ�Ʈ
        if (!trackerDemo.getIsTracking()) {
            cv::putText(display, "Draw a box around the object to track",
                cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX,
                0.8, cv::Scalar(255, 255, 0), 2);
        }

        // ���� Ʈ��Ŀ ����
        cv::putText(display, "Current: " + trackerDemo.trackerType,
            cv::Point(20, display.rows - 20), cv::FONT_HERSHEY_SIMPLEX,
            0.7, cv::Scalar(200, 200, 200), 2);

        cv::imshow("Tracker Demo", display);

        char key = cv::waitKey(1);

        // Ű���� �Է� ó��
        if (key == 27) break; // ESC
        else if (key == ' ') { // SPACE - �ʱ�ȭ
            trackerDemo.createTracker(trackerDemo.trackerType);
            selectedBbox = cv::Rect2d();
        }
        else if (key >= '1' && key <= '7') { // Ʈ��Ŀ ����
            std::vector<std::string> trackerTypes = {
                "CSRT", "KCF", "MOSSE", "MIL", "BOOSTING", "MEDIANFLOW", "TLD"
            };
            int index = key - '1';
            if (index < trackerTypes.size()) {
                trackerDemo.createTracker(trackerTypes[index]);
                std::cout << "Ʈ��Ŀ ����: " << trackerTypes[index] << std::endl;
            }
        }

        // ���� �Ϸ� �� Ʈ��ŷ ����
        if (!selectObject && selectedBbox.width > 0 && selectedBbox.height > 0
            && !trackerDemo.getIsTracking()) {
            trackerDemo.initTracking(frame, selectedBbox);
            selectedBbox = cv::Rect2d(); // ����
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}