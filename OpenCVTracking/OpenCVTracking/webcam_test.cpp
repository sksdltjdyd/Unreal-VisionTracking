// webcam_test.cpp
// OpenCV ��ġ Ȯ�� �� �⺻ ��ķ �׽�Ʈ

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

int main() {
    std::cout << "=== OpenCV ��ķ �׽�Ʈ ===" << std::endl;
    std::cout << "OpenCV ����: " << CV_VERSION << std::endl;

    // ��ķ ���� (0�� = �⺻ ��ķ)
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "Error: ��ķ�� �� �� �����ϴ�!" << std::endl;
        std::cerr << "��ķ�� ����Ǿ� �ִ��� Ȯ���ϼ���." << std::endl;
        return -1;
    }

    // ��ķ ����
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    cap.set(cv::CAP_PROP_FPS, 30);

    // ���� ������ Ȯ��
    double width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    double height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);

    std::cout << "��ķ �ػ�: " << width << "x" << height << std::endl;
    std::cout << "FPS: " << fps << std::endl;
    std::cout << "\nESC Ű�� ������ �����մϴ�." << std::endl;

    cv::Mat frame;
    cv::namedWindow("Webcam Test", cv::WINDOW_NORMAL);

    // FPS ������ ����
    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    double measured_fps = 0.0;

    while (true) {
        // ������ ĸó
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Warning: �� ������!" << std::endl;
            continue;
        }

        // FPS ���
        frame_count++;
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();

        if (elapsed > 1000) {  // 1�ʸ��� FPS ������Ʈ
            measured_fps = frame_count * 1000.0 / elapsed;
            frame_count = 0;
            start_time = current_time;
        }

        // ȭ�鿡 ���� ǥ��
        cv::putText(frame, "OpenCV Test - Press ESC to exit",
            cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
            0.7, cv::Scalar(0, 255, 0), 2);

        cv::putText(frame, "FPS: " + std::to_string(measured_fps),
            cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX,
            0.7, cv::Scalar(0, 255, 0), 2);

        // ������ ��� ���� �׽�Ʈ (�߾ӿ� �� �׸���)
        cv::circle(frame, cv::Point(width / 2, height / 2), 50,
            cv::Scalar(0, 0, 255), 2);

        cv::putText(frame, "Target Area",
            cv::Point(width / 2 - 40, height / 2 + 80),
            cv::FONT_HERSHEY_SIMPLEX, 0.5,
            cv::Scalar(0, 0, 255), 1);

        // ������ ǥ��
        cv::imshow("Webcam Test", frame);

        // ESC Ű Ȯ�� (27 = ESC)
        if (cv::waitKey(1) == 27) {
            std::cout << "\nESC Ű�� ���Ƚ��ϴ�. �����մϴ�." << std::endl;
            break;
        }
    }

    // ����
    cap.release();
    cv::destroyAllWindows();

    std::cout << "�׽�Ʈ �Ϸ�!" << std::endl;
    return 0;
}