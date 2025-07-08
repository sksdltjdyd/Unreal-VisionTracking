// color_pong_game_kalman.cpp
// 칼만 필터로 부드러운 추적이 가능한 색상 추적 Pong 게임

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <memory>

const int GAME_WIDTH = 1280;
const int GAME_HEIGHT = 720;
const int PADDLE_WIDTH = 20;
const int PADDLE_HEIGHT = 100;
const int BALL_SIZE = 20;
const float BALL_SPEED = 5.0f;
const int WINNING_SCORE = 5;

// 칼만 필터 트래커 클래스
class KalmanTracker {
private:
    cv::KalmanFilter kf;
    cv::Mat state;
    cv::Mat measurement;
    bool initialized;
    int lostFrames;
    const int MAX_LOST_FRAMES = 15; // Pong 게임용으로 약간 늘림

public:
    cv::Point2f lastPosition;
    cv::Point2f velocity;
    std::deque<cv::Point2f> trajectory;

    KalmanTracker() : initialized(false), lostFrames(0) {
        kf.init(4, 2, 0);

        // 상태 전이 행렬
        kf.transitionMatrix = (cv::Mat_<float>(4, 4) <<
            1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1);

        // 관측 행렬
        kf.measurementMatrix = (cv::Mat_<float>(2, 4) <<
            1, 0, 0, 0,
            0, 1, 0, 0);

        // 노이즈 설정 (게임용으로 조정)
        cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-3));
        cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-2));
        cv::setIdentity(kf.errorCovPost, cv::Scalar::all(.1));

        state = cv::Mat::zeros(4, 1, CV_32F);
        measurement = cv::Mat::zeros(2, 1, CV_32F);
    }

    void initializeTracker(const cv::Point2f& pt) {
        state.at<float>(0) = pt.x;
        state.at<float>(1) = pt.y;
        state.at<float>(2) = 0;
        state.at<float>(3) = 0;

        kf.statePre = state.clone();
        kf.statePost = state.clone();

        lastPosition = pt;
        initialized = true;
        lostFrames = 0;

        trajectory.clear();
        trajectory.push_back(pt);
    }

    cv::Point2f predict() {
        if (!initialized) return cv::Point2f(-1, -1);

        cv::Mat prediction = kf.predict();
        cv::Point2f predictedPt(prediction.at<float>(0), prediction.at<float>(1));
        velocity.x = prediction.at<float>(2);
        velocity.y = prediction.at<float>(3);

        return predictedPt;
    }

    cv::Point2f update(const cv::Point2f& measuredPt, bool found) {
        if (!initialized) {
            if (found) initializeTracker(measuredPt);
            return measuredPt;
        }

        cv::Point2f predictedPt = predict();

        if (found) {
            measurement.at<float>(0) = measuredPt.x;
            measurement.at<float>(1) = measuredPt.y;

            cv::Mat corrected = kf.correct(measurement);

            lastPosition.x = corrected.at<float>(0);
            lastPosition.y = corrected.at<float>(1);

            lostFrames = 0;

            trajectory.push_back(lastPosition);
            if (trajectory.size() > 30) trajectory.pop_front();
        }
        else {
            lostFrames++;
            lastPosition = predictedPt;

            if (lostFrames > MAX_LOST_FRAMES) {
                initialized = false;
                trajectory.clear();
            }
        }

        return lastPosition;
    }

    bool isTracking() const {
        return initialized && lostFrames < MAX_LOST_FRAMES;
    }
};

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

class ColorPongGameKalman {
private:
    Paddle player1;
    Paddle player2;
    Ball ball;

    bool gameRunning;
    std::string winner;

    // 칼만 트래커 추가
    std::unique_ptr<KalmanTracker> player1Tracker;
    std::unique_ptr<KalmanTracker> player2Tracker;

    // 디버그 표시 옵션
    bool showTrajectory;
    bool showPrediction;

public:
    int p1_h_min, p1_h_max, p1_s_min, p1_s_max, p1_v_min, p1_v_max;
    int p2_h_min, p2_h_max, p2_s_min, p2_s_max, p2_v_min, p2_v_max;

    cv::Mat player1_mask;
    cv::Mat player2_mask;

    ColorPongGameKalman()
        : player1(50, GAME_HEIGHT / 2 - PADDLE_HEIGHT / 2, cv::Scalar(0, 0, 255)),
        player2(GAME_WIDTH - 50 - PADDLE_WIDTH, GAME_HEIGHT / 2 - PADDLE_HEIGHT / 2, cv::Scalar(0, 255, 255)),
        gameRunning(true), showTrajectory(true), showPrediction(true) {

        // 칼만 트래커 초기화
        player1Tracker = std::make_unique<KalmanTracker>();
        player2Tracker = std::make_unique<KalmanTracker>();

        // HSV 기본값
        p1_h_min = 0;   p1_h_max = 10;
        p1_s_min = 120; p1_s_max = 255;
        p1_v_min = 100; p1_v_max = 255;

        p2_h_min = 20;  p2_h_max = 40;
        p2_s_min = 120; p2_s_max = 255;
        p2_v_min = 100; p2_v_max = 255;
    }

    void updateTracking(const cv::Mat& frame) {
        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        cv::Scalar player1_lower(p1_h_min, p1_s_min, p1_v_min);
        cv::Scalar player1_upper(p1_h_max, p1_s_max, p1_v_max);

        cv::Scalar player2_lower(p2_h_min, p2_s_min, p2_v_min);
        cv::Scalar player2_upper(p2_h_max, p2_s_max, p2_v_max);

        // Player 1 추적 (칼만 필터 적용)
        cv::Point2f player1_raw = detectColor(hsv, player1_lower, player1_upper, player1_mask, true);
        bool p1_found = (player1_raw.x > 0);
        cv::Point2f player1_filtered = player1Tracker->update(player1_raw, p1_found);

        if (player1Tracker->isTracking()) {
            player1.updatePosition(player1_filtered.y);
        }

        // Player 2 추적 (칼만 필터 적용)
        cv::Point2f player2_raw = detectColor(hsv, player2_lower, player2_upper, player2_mask, false);
        bool p2_found = (player2_raw.x > 0);
        cv::Point2f player2_filtered = player2Tracker->update(player2_raw, p2_found);

        if (player2Tracker->isTracking()) {
            player2.updatePosition(player2_filtered.y);
        }
    }

    cv::Point detectColor(const cv::Mat& hsv, const cv::Scalar& lower, const cv::Scalar& upper, cv::Mat& out_mask, bool is_red) {
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
                if (area > maxArea && area > 1000) {
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
        float ballLeft = ball.position.x - ball.radius;
        float ballRight = ball.position.x + ball.radius;
        float ballTop = ball.position.y - ball.radius;
        float ballBottom = ball.position.y + ball.radius;

        float paddleLeft = paddle.rect.x;
        float paddleRight = paddle.rect.x + paddle.rect.width;
        float paddleTop = paddle.rect.y;
        float paddleBottom = paddle.rect.y + paddle.rect.height;

        if (ballRight >= paddleLeft && ballLeft <= paddleRight &&
            ballBottom >= paddleTop && ballTop <= paddleBottom) {

            if (paddle.rect.x < GAME_WIDTH / 2) {
                if (ball.velocity.x < 0) {
                    ball.velocity.x = -ball.velocity.x;
                    float paddleCenter = paddleTop + paddle.rect.height / 2.0f;
                    float hitPosition = (ball.position.y - paddleCenter) / (paddle.rect.height / 2.0f);
                    ball.velocity.y = hitPosition * BALL_SPEED * 0.75f;
                    ball.position.x = paddleRight + ball.radius;
                }
            }
            else {
                if (ball.velocity.x > 0) {
                    ball.velocity.x = -ball.velocity.x;
                    float paddleCenter = paddleTop + paddle.rect.height / 2.0f;
                    float hitPosition = (ball.position.y - paddleCenter) / (paddle.rect.height / 2.0f);
                    ball.velocity.y = hitPosition * BALL_SPEED * 0.75f;
                    ball.position.x = paddleLeft - ball.radius;
                }
            }

            float speed = sqrt(ball.velocity.x * ball.velocity.x + ball.velocity.y * ball.velocity.y);
            if (speed > BALL_SPEED * 1.5f) {
                ball.velocity.x = (ball.velocity.x / speed) * BALL_SPEED * 1.5f;
                ball.velocity.y = (ball.velocity.y / speed) * BALL_SPEED * 1.5f;
            }
        }
    }

    void checkWinner() {
        if (player1.score >= WINNING_SCORE) {
            winner = "Red Player Wins!";
            gameRunning = false;
        }
        else if (player2.score >= WINNING_SCORE) {
            winner = "Yellow Player Wins!";
            gameRunning = false;
        }
    }

    void draw(cv::Mat& frame) {
        frame = cv::Scalar(0, 0, 0);

        // 칼만 필터 디버그 정보 표시
        if (showTrajectory) {
            // Player 1 궤적 (반투명 빨강)
            for (size_t i = 1; i < player1Tracker->trajectory.size(); i++) {
                cv::line(frame,
                    cv::Point(50, player1Tracker->trajectory[i - 1].y),
                    cv::Point(50, player1Tracker->trajectory[i].y),
                    cv::Scalar(0, 0, 128), 2);
            }

            // Player 2 궤적 (반투명 노랑)
            for (size_t i = 1; i < player2Tracker->trajectory.size(); i++) {
                cv::line(frame,
                    cv::Point(GAME_WIDTH - 50, player2Tracker->trajectory[i - 1].y),
                    cv::Point(GAME_WIDTH - 50, player2Tracker->trajectory[i].y),
                    cv::Scalar(0, 128, 128), 2);
            }
        }

        // 중앙선
        for (int y = 0; y < GAME_HEIGHT; y += 20) {
            cv::line(frame, cv::Point(GAME_WIDTH / 2, y), cv::Point(GAME_WIDTH / 2, y + 10),
                cv::Scalar(100, 100, 100), 2);
        }

        // 패들
        cv::rectangle(frame, player1.rect, player1.color, -1);
        cv::rectangle(frame, player2.rect, player2.color, -1);

        // 예측 표시
        if (showPrediction && !player1Tracker->trajectory.empty()) {
            // Player 1 예측 위치
            cv::Point predictedP1(50, player1Tracker->lastPosition.y + player1Tracker->velocity.y * 10);
            cv::circle(frame, predictedP1, 5, cv::Scalar(255, 100, 100), 1);

            // Player 2 예측 위치
            cv::Point predictedP2(GAME_WIDTH - 50, player2Tracker->lastPosition.y + player2Tracker->velocity.y * 10);
            cv::circle(frame, predictedP2, 5, cv::Scalar(100, 255, 255), 1);
        }

        // 공
        cv::circle(frame, cv::Point(ball.position), ball.radius, cv::Scalar(255, 255, 255), -1);

        // 점수
        cv::putText(frame, std::to_string(player1.score), cv::Point(GAME_WIDTH / 2 - 100, 80),
            cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 255, 255), 3);
        cv::putText(frame, std::to_string(player2.score), cv::Point(GAME_WIDTH / 2 + 50, 80),
            cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 255, 255), 3);

        // 안내 문구
        cv::putText(frame, "Kalman Filter ON - Red vs Yellow", cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200, 200, 200), 2);
        cv::putText(frame, "T: Toggle Trajectory, P: Toggle Prediction", cv::Point(10, 55),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(150, 150, 150), 1);

        // 게임 종료 메시지
        if (!gameRunning) {
            cv::Mat overlay = frame.clone();
            cv::rectangle(overlay, cv::Point(0, 0), cv::Point(GAME_WIDTH, GAME_HEIGHT),
                cv::Scalar(0, 0, 0), -1);
            cv::addWeighted(overlay, 0.7, frame, 0.3, 0, frame);

            cv::Size textSize = cv::getTextSize(winner, cv::FONT_HERSHEY_SIMPLEX, 2, 3, nullptr);
            cv::Point textPos((GAME_WIDTH - textSize.width) / 2, GAME_HEIGHT / 2);
            cv::putText(frame, winner, textPos, cv::FONT_HERSHEY_SIMPLEX, 2,
                cv::Scalar(255, 255, 255), 3);

            std::string restartMsg = "Press SPACE to restart";
            textSize = cv::getTextSize(restartMsg, cv::FONT_HERSHEY_SIMPLEX, 1, 2, nullptr);
            textPos = cv::Point((GAME_WIDTH - textSize.width) / 2, GAME_HEIGHT / 2 + 50);
            cv::putText(frame, restartMsg, textPos, cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(200, 200, 200), 2);
        }
    }

    void restart() {
        player1.score = 0;
        player2.score = 0;
        ball.reset();
        gameRunning = true;
        winner.clear();

        // 칼만 트래커 리셋
        player1Tracker = std::make_unique<KalmanTracker>();
        player2Tracker = std::make_unique<KalmanTracker>();
    }

    void toggleTrajectory() { showTrajectory = !showTrajectory; }
    void togglePrediction() { showPrediction = !showPrediction; }
    bool isRunning() const { return gameRunning; }
};

int main() {
    std::cout << "=== Kalman Filter Pong Game ===" << std::endl;
    std::cout << "칼만 필터로 부드러운 추적이 가능합니다!" << std::endl;
    std::cout << "Player 1 (Left): Red Object" << std::endl;
    std::cout << "Player 2 (Right): Yellow Object" << std::endl;
    std::cout << "T: 궤적 표시 ON/OFF" << std::endl;
    std::cout << "P: 예측 표시 ON/OFF" << std::endl;
    std::cout << "SPACE: Restart, ESC: Exit" << std::endl << std::endl;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Could not open webcam!" << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, GAME_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, GAME_HEIGHT);

    ColorPongGameKalman game;

    cv::namedWindow("Kalman Pong", cv::WINDOW_NORMAL);
    cv::namedWindow("Camera View", cv::WINDOW_NORMAL);
    cv::namedWindow("HSV Controls", cv::WINDOW_NORMAL);
    cv::namedWindow("Red Mask", cv::WINDOW_NORMAL);
    cv::namedWindow("Yellow Mask", cv::WINDOW_NORMAL);

    cv::resizeWindow("Camera View", 480, 270);
    cv::resizeWindow("Red Mask", 480, 270);
    cv::resizeWindow("Yellow Mask", 480, 270);

    // HSV 트랙바 생성
    cv::createTrackbar("P1 H_MIN", "HSV Controls", &game.p1_h_min, 180);
    cv::createTrackbar("P1 H_MAX", "HSV Controls", &game.p1_h_max, 180);
    cv::createTrackbar("P1 S_MIN", "HSV Controls", &game.p1_s_min, 255);
    cv::createTrackbar("P1 S_MAX", "HSV Controls", &game.p1_s_max, 255);
    cv::createTrackbar("P1 V_MIN", "HSV Controls", &game.p1_v_min, 255);
    cv::createTrackbar("P1 V_MAX", "HSV Controls", &game.p1_v_max, 255);

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

        cv::imshow("Kalman Pong", gameFrame);
        cv::imshow("Camera View", frame);
        cv::imshow("Red Mask", game.player1_mask);
        cv::imshow("Yellow Mask", game.player2_mask);

        char key = cv::waitKey(16);
        if (key == 27) break;
        else if (key == ' ' && !game.isRunning()) {
            game.restart();
        }
        else if (key == 't' || key == 'T') {
            game.toggleTrajectory();
        }
        else if (key == 'p' || key == 'P') {
            game.togglePrediction();
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}