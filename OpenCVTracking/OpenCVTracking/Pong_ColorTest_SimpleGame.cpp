// color_pong_game_revised.cpp
// HSV 조절 및 마스크 표시 기능이 추가된 색상 추적 Pong 게임 (Red vs Yellow)

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

// ... (게임 상수 및 구조체 정의는 변경 없음) ...
const int GAME_WIDTH = 1280;
const int GAME_HEIGHT = 720;
const int PADDLE_WIDTH = 20;
const int PADDLE_HEIGHT = 100;
const int BALL_SIZE = 20;
const float BALL_SPEED = 5.0f;
const int WINNING_SCORE = 5;

struct Paddle {
    cv::Rect rect;
    cv::Scalar color;
    int score;

    Paddle(int x, int y, cv::Scalar c)
        : rect(x, y, PADDLE_WIDTH, PADDLE_HEIGHT), color(c), score(0) {
    }

    void updatePosition(int y) {
        rect.y = std::max(0, std::min(y - PADDLE_HEIGHT / 2, GAME_HEIGHT - PADDLE_HEIGHT));
    }
};

struct Ball {
    cv::Point2f position;
    cv::Point2f velocity;
    int radius;

    Ball() : position(GAME_WIDTH / 2, GAME_HEIGHT / 2),
        velocity(BALL_SPEED, BALL_SPEED), radius(BALL_SIZE / 2) {
        float angle = (rand() % 60 - 30) * CV_PI / 180.0f;
        velocity.x = BALL_SPEED * cos(angle) * (rand() % 2 ? 1 : -1);
        velocity.y = BALL_SPEED * sin(angle);
    }

    void update() {
        position += velocity;
        if (position.y <= radius || position.y >= GAME_HEIGHT - radius) {
            velocity.y = -velocity.y;
        }
    }

    void reset() {
        position = cv::Point2f(GAME_WIDTH / 2, GAME_HEIGHT / 2);
        float angle = (rand() % 60 - 30) * CV_PI / 180.0f;
        velocity.x = BALL_SPEED * cos(angle) * (rand() % 2 ? 1 : -1);
        velocity.y = BALL_SPEED * sin(angle);
    }
};


class ColorPongGame {
private:
    Paddle player1;
    Paddle player2;
    Ball ball;

    bool gameRunning;
    std::string winner;

public:
    // [추가] 트랙바와 연동하기 위해 HSV 값을 int 멤버 변수로 변경
    int p1_h_min, p1_h_max, p1_s_min, p1_s_max, p1_v_min, p1_v_max;
    int p2_h_min, p2_h_max, p2_s_min, p2_s_max, p2_v_min, p2_v_max;

    // [추가] 마스크를 외부에 표시하기 위한 Mat 멤버
    cv::Mat player1_mask;
    cv::Mat player2_mask;

    ColorPongGame()
        // [수정] Player 2의 패들 색상을 노란색으로 변경
        : player1(50, GAME_HEIGHT / 2 - PADDLE_HEIGHT / 2, cv::Scalar(0, 0, 255)), // Red
        player2(GAME_WIDTH - 50 - PADDLE_WIDTH, GAME_HEIGHT / 2 - PADDLE_HEIGHT / 2, cv::Scalar(0, 255, 255)), // Yellow
        gameRunning(true) {

        // [수정] Player 1 (빨간색) HSV 기본값
        // 빨간색은 H값이 0 주변과 180 주변에 모두 있으므로, 낮은 쪽을 기준으로 설정
        p1_h_min = 0;   p1_h_max = 10;
        p1_s_min = 120; p1_s_max = 255;
        p1_v_min = 100; p1_v_max = 255;

        // [수정] Player 2 (노란색) HSV 기본값
        p2_h_min = 20;  p2_h_max = 40;
        p2_s_min = 120; p2_s_max = 255;
        p2_v_min = 100; p2_v_max = 255;
    }

    void updateTracking(const cv::Mat& frame) {
        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        // [수정] 트랙바에서 조절된 값을 사용하여 색상 범위 업데이트
        cv::Scalar player1_lower(p1_h_min, p1_s_min, p1_v_min);
        cv::Scalar player1_upper(p1_h_max, p1_s_max, p1_v_max);

        cv::Scalar player2_lower(p2_h_min, p2_s_min, p2_v_min);
        cv::Scalar player2_upper(p2_h_max, p2_s_max, p2_v_max);

        // Player 1 (빨간색) 추적 및 마스크 저장
        cv::Point player1_pos = trackColor(hsv, player1_lower, player1_upper, player1_mask, true);
        if (player1_pos.x > 0) {
            player1.updatePosition(player1_pos.y);
        }

        // Player 2 (노란색) 추적 및 마스크 저장
        cv::Point player2_pos = trackColor(hsv, player2_lower, player2_upper, player2_mask, false);
        if (player2_pos.x > 0) {
            player2.updatePosition(player2_pos.y);
        }
    }

    // [수정] 마스크를 출력하고, 빨간색을 특별 처리하는 로직 추가
    cv::Point trackColor(const cv::Mat& hsv, const cv::Scalar& lower, const cv::Scalar& upper, cv::Mat& out_mask, bool is_red) {
        // 빨간색은 H값이 0과 180 근처에 걸쳐 있으므로 두 범위를 합쳐야 함
        if (is_red && lower[0] < 10) {
            cv::Mat mask1, mask2;
            cv::inRange(hsv, lower, upper, mask1);
            cv::inRange(hsv, cv::Scalar(170, p1_s_min, p1_v_min), cv::Scalar(180, p1_s_max, p1_v_max), mask2);
            out_mask = mask1 | mask2;
        }
        else {
            cv::inRange(hsv, lower, upper, out_mask);
        }

        cv::erode(out_mask, out_mask, cv::Mat(), cv::Point(-1, -1), 2);
        cv::dilate(out_mask, out_mask, cv::Mat(), cv::Point(-1, -1), 2);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(out_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if (!contours.empty()) {
            double maxArea = 0;
            int maxIdx = -1;

            for (int i = 0; i < contours.size(); i++) {
                double area = cv::contourArea(contours[i]);
                if (area > maxArea && area > 1000) { // 최소 면적 필터
                    maxArea = area;
                    maxIdx = i;
                }
            }

            if (maxIdx >= 0) {
                cv::Moments m = cv::moments(contours[maxIdx]);
                if (m.m00 != 0) {
                    return cv::Point(int(m.m10 / m.m00), int(m.m01 / m.m00));
                }
            }
        }

        return cv::Point(-1, -1);
    }

    void updateGame() {
        if (!gameRunning) return;
        ball.update();
        checkPaddleCollision(player1);
        checkPaddleCollision(player2);

        if (ball.position.x <= 0) {
            player2.score++;
            ball.reset();
            checkWinner();
        }
        else if (ball.position.x >= GAME_WIDTH) {
            player1.score++;
            ball.reset();
            checkWinner();
        }
    }

    void checkPaddleCollision(const Paddle& paddle) {
        // ... (충돌 로직 변경 없음) ...
    }

    void checkWinner() {
        if (player1.score >= WINNING_SCORE) {
            winner = "Red Player Wins!";
            gameRunning = false;
        }
        else if (player2.score >= WINNING_SCORE) {
            // [수정] 승리 메시지 변경
            winner = "Yellow Player Wins!";
            gameRunning = false;
        }
    }

    void draw(cv::Mat& frame) {
        frame = cv::Scalar(0, 0, 0);

        // ... (중앙선, 패들, 공, 점수 그리기 로직 변경 없음) ...
        for (int y = 0; y < GAME_HEIGHT; y += 20) {
            cv::line(frame, cv::Point(GAME_WIDTH / 2, y), cv::Point(GAME_WIDTH / 2, y + 10), cv::Scalar(100, 100, 100), 2);
        }
        cv::rectangle(frame, player1.rect, player1.color, -1);
        cv::rectangle(frame, player2.rect, player2.color, -1);
        cv::circle(frame, cv::Point(ball.position), ball.radius, cv::Scalar(255, 255, 255), -1);
        cv::putText(frame, std::to_string(player1.score), cv::Point(GAME_WIDTH / 2 - 100, 80), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 255, 255), 3);
        cv::putText(frame, std::to_string(player2.score), cv::Point(GAME_WIDTH / 2 + 50, 80), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 255, 255), 3);

        // [수정] 안내 문구 변경
        cv::putText(frame, "Red vs Yellow - First to " + std::to_string(WINNING_SCORE), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200, 200, 200), 2);

        if (!gameRunning) {
            // ... (게임 종료 메시지 그리기 로직 변경 없음) ...
        }
    }

    void restart() {
        // ... (재시작 로직 변경 없음) ...
    }

    bool isRunning() const { return gameRunning; }
};

int main() {
    std::cout << "=== Color Tracking Pong Game (Revised) ===" << std::endl;
    // [수정] 안내 문구 변경
    std::cout << "Player 1 (Left): Red Object" << std::endl;
    std::cout << "Player 2 (Right): Yellow Object" << std::endl;
    std::cout << "First to " << WINNING_SCORE << " points wins!" << std::endl;
    std::cout << "SPACE: Restart, ESC: Exit" << std::endl << std::endl;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Could not open webcam!" << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, GAME_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, GAME_HEIGHT);

    ColorPongGame game;

    // [추가] 컨트롤 및 마스크를 위한 창 생성
    cv::namedWindow("Color Pong", cv::WINDOW_NORMAL);
    cv::namedWindow("Camera View", cv::WINDOW_NORMAL);
    cv::namedWindow("HSV Controls", cv::WINDOW_NORMAL);
    cv::namedWindow("Red Mask", cv::WINDOW_NORMAL);
    cv::namedWindow("Yellow Mask", cv::WINDOW_NORMAL);

    cv::resizeWindow("Camera View", 480, 270);
    cv::resizeWindow("Red Mask", 480, 270);
    cv::resizeWindow("Yellow Mask", 480, 270);

    // [추가] Player 1 (Red) HSV 트랙바 생성
    cv::createTrackbar("P1 H_MIN", "HSV Controls", &game.p1_h_min, 180);
    cv::createTrackbar("P1 H_MAX", "HSV Controls", &game.p1_h_max, 180);
    cv::createTrackbar("P1 S_MIN", "HSV Controls", &game.p1_s_min, 255);
    cv::createTrackbar("P1 S_MAX", "HSV Controls", &game.p1_s_max, 255);
    cv::createTrackbar("P1 V_MIN", "HSV Controls", &game.p1_v_min, 255);
    cv::createTrackbar("P1 V_MAX", "HSV Controls", &game.p1_v_max, 255);

    // [추가] Player 2 (Yellow) HSV 트랙바 생성
    cv::createTrackbar("P2 H_MIN", "HSV Controls", &game.p2_h_min, 180);
    cv::createTrackbar("P2 H_MAX", "HSV Controls", &game.p2_h_max, 180);
    cv::createTrackbar("P2 S_MIN", "HSV Controls", &game.p2_s_min, 255);
    cv::createTrackbar("P2 S_MAX", "HSV Controls", &game.p2_s_max, 255);
    cv::createTrackbar("P2 V_MIN", "HSV Controls", &game.p2_v_min, 255);
    cv::createTrackbar("P2 V_MAX", "HSV Controls", &game.p2_v_max, 255);

    cv::Mat frame, gameFrame(GAME_HEIGHT, GAME_WIDTH, CV_8UC3);

    while (true) {
        cap >> frame;
        if (frame.empty()) continue;

        cv::flip(frame, frame, 1);

        game.updateTracking(frame);
        game.updateGame();
        game.draw(gameFrame);

        cv::imshow("Color Pong", gameFrame);
        cv::imshow("Camera View", frame);

        // [추가] 마스크 창 표시
        cv::imshow("Red Mask", game.player1_mask);
        cv::imshow("Yellow Mask", game.player2_mask);

        char key = cv::waitKey(16); // ~60 FPS
        if (key == 27) break;
        else if (key == ' ' && !game.isRunning()) {
            game.restart();
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}