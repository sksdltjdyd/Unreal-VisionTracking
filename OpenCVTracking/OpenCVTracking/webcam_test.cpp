// webcam_test.cpp
// OpenCV 설치 확인 및 기본 웹캠 테스트

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

int main() {
    std::cout << "=== OpenCV 웹캠 테스트 ===" << std::endl;
    std::cout << "OpenCV 버전: " << CV_VERSION << std::endl;

    // 웹캠 열기 (0번 = 기본 웹캠)
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "Error: 웹캠을 열 수 없습니다!" << std::endl;
        std::cerr << "웹캠이 연결되어 있는지 확인하세요." << std::endl;
        return -1;
    }

    // 웹캠 설정
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    cap.set(cv::CAP_PROP_FPS, 30);

    // 실제 설정값 확인
    double width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);

    std::cout << "웹캠 해상도: " << width << "x" << height << std::endl;
    std::cout << "FPS: " << fps << std::endl;
    std::cout << "\nESC 키를 누르면 종료합니다." << std::endl;

    cv::Mat frame;
    cv::namedWindow("Webcam Test", cv::WINDOW_NORMAL);

    // FPS 측정용 변수
    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    double measured_fps = 0.0;

    while (true) {
        // 프레임 캡처
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Warning: 빈 프레임!" << std::endl;
            continue;
        }

        // FPS 계산
        frame_count++;
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();

        if (elapsed > 1000) {  // 1초마다 FPS 업데이트
            measured_fps = frame_count * 1000.0 / elapsed;
            frame_count = 0;
            start_time = current_time;
        }

        // 화면에 정보 표시
        cv::putText(frame, "OpenCV Test - Press ESC to exit",
            cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
            0.7, cv::Scalar(0, 255, 0), 2);

        cv::putText(frame, "FPS: " + std::to_string(measured_fps),
            cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX,
            0.7, cv::Scalar(0, 255, 0), 2);

        // 간단한 모션 감지 테스트 (중앙에 원 그리기)
        cv::circle(frame, cv::Point(width / 2, height / 2), 50,
            cv::Scalar(0, 0, 255), 2);

        cv::putText(frame, "Target Area",
            cv::Point(width / 2 - 40, height / 2 + 80),
            cv::FONT_HERSHEY_SIMPLEX, 0.5,
            cv::Scalar(0, 0, 255), 1);

        // 프레임 표시
        cv::imshow("Webcam Test", frame);

        // ESC 키 확인 (27 = ESC)
        if (cv::waitKey(1) == 27) {
            std::cout << "\nESC 키가 눌렸습니다. 종료합니다." << std::endl;
            break;
        }
    }

    // 정리
    cap.release();
    cv::destroyAllWindows();

    std::cout << "테스트 완료!" << std::endl;
    return 0;
}