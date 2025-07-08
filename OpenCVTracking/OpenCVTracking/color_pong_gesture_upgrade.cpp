// color_pong_gesture.cpp
// 궤적 그리기와 제스처 인식이 추가된 색상 추적 Pong 게임

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <memory>
#include <deque>
#include <algorithm>

const int GAME_WIDTH = 1280;
const int GAME_HEIGHT = 720;
const int PADDLE_WIDTH = 20;
const int PADDLE_HEIGHT = 100;
const int BALL_SIZE = 20;
const float BALL_SPEED = 5.0f;
const int WINNING_SCORE = 5;

// 제스처 타입
enum class GestureType {
    NONE,
    CIRCLE,      // 원 그리기 - 슈퍼샷
    ZIGZAG,      // 지그재그 - 방어막
    VERTICAL,    // 수직선 - 속도 증가
    HORIZONTAL   // 수평선 - 공 느리게
};

// 궤적 분석기
class TrajectoryAnalyzer {
private:
    std::deque<cv::Point2f> points;
    const int MAX_POINTS = 30;
    const float MIN_GESTURE_DISTANCE = 100.0f;

public:
    void addPoint(const cv::Point2f& pt) {
        points.push_back(pt);
        if (points.size() > MAX_POINTS) {
            points.pop_front();
        }
    }

    void clear() {
        points.clear();
    }

    float getTotalDistance() const {
        float dist = 0;
        for (size_t i = 1; i < points.size(); i++) {
            dist += cv::norm(points[i] - points[i - 1]);
        }
        return dist;
    }

    GestureType detectGesture() {
        if (points.size() < 20) return GestureType::NONE;
        if (getTotalDistance() < MIN_GESTURE_DISTANCE) return GestureType::NONE;

        // 원 감지
        if (isCircle()) return GestureType::CIRCLE;

        // 지그재그 감지
        if (isZigzag()) return GestureType::ZIGZAG;

        // 수직/수평선 감지
        cv::Point2f start = points.front();
        cv::Point2f end = points.back();
        float dx = abs(end.x - start.x);
        float dy = abs(end.y - start.y);

        if (dy > dx * 3 && dy > 150) return GestureType::VERTICAL;
        if (dx > dy * 3 && dx > 150) return GestureType::HORIZONTAL;

        return GestureType::NONE;
    }

    bool isCircle() {
        if (points.size() < 20) return false;

        // 시작점과 끝점 거리
        float endDist = cv::norm(points.back() - points.front());
        if (endDist > 50) return false;

        // 중심점 계산
        cv::Point2f center(0, 0);
        for (const auto& pt : points) {
            center += pt;
        }
        center *= (1.0f / points.size());

        // 반지름 계산
        float avgRadius = 0;
        for (const auto& pt : points) {
            avgRadius += cv::norm(pt - center);
        }
        avgRadius /= points.size();

        // 원형도 검사
        float variance = 0;
        for (const auto& pt : points) {
            float r = cv::norm(pt - center);
            variance += (r - avgRadius) * (r - avgRadius);
        }
        variance /= points.size();

        return variance < avgRadius * avgRadius * 0.3f && avgRadius > 30;
    }

    bool isZigzag() {
        if (points.size() < 15) return false;

        int directionChanges = 0;
        float lastDx = 0;

        for (size_t i = 2; i < points.size(); i++) {
            float dx = points[i].x - points[i - 1].x;
            if (lastDx * dx < 0 && abs(dx) > 20) {
                directionChanges++;
            }
            if (dx != 0) lastDx = dx;
        }

        return directionChanges >= 3;
    }

    const std::deque<cv::Point2f>& getPoints() const { return points; }
};

// 파티클 효과
struct Particle {
    cv::Point2f position;
    cv::Point2f velocity;
    cv::Scalar color;
    float life;

    Particle(cv::Point2f pos, cv::Point2f vel, cv::Scalar col)
        : position(pos), velocity(vel), color(col), life(1.0f) {
    }

    void update() {
        position += velocity;
        velocity *= 0.98f;
        life -= 0.02f;
    }

    bool isDead() const { return life <= 0; }
};

// 파워업 효과
struct PowerUp {
    GestureType type;
    float duration;
    float maxDuration;
    std::string name;
    cv::Scalar color;

    PowerUp(GestureType t, float dur, const std::string& n, cv::Scalar c)
        : type(t), duration(dur), maxDuration(dur), name(n), color(c) {
    }

    void update(float dt) {
        duration -= dt;
    }

    bool isActive() const { return duration > 0; }
    float getProgress() const { return duration / maxDuration; }
};

// 칼만 필터는 이전 코드와 동일
class KalmanTracker {
    // ... (이전 코드와 동일)
private:
    cv::KalmanFilter kf;
    cv::Mat state;
    cv::Mat measurement;
    bool initialized;
    int lostFrames;
    const int MAX_LOST_FRAMES = 15;

public:
    cv::Point2f lastPosition;
    cv::Point2f velocity;
    std::deque<cv::Point2f> trajectory;

    KalmanTracker() : initialized(false), lostFrames(0) {
        kf.init(4, 2, 0);

        kf.transitionMatrix = (cv::Mat_<float>(4, 4) <<
            1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 0, 1,
            0, 0, 0, 1);

        kf.measurementMatrix = (cv::Mat_<float>(2, 4) <<
            1, 0, 0, 0,
            0, 1, 0, 0);

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

    cv::Point2f update(const cv::Point2f& measuredPt, bool found) {
        if (!initialized) {
            if (found) initializeTracker(measuredPt);
            return measuredPt;
        }

        cv::Mat prediction = kf.predict();
        cv::Point2f predictedPt(prediction.at<float>(0), prediction.at<float>(1));
        velocity.x = prediction.at<float>(2);
        velocity.y = prediction.at<float>(3);

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
    bool hasShield;

    Paddle(int x, int y, cv::Scalar c)
        : rect(x, y, PADDLE_WIDTH, PADDLE_HEIGHT), color(c), score(0), hasShield(false) {
    }

    void updatePosition(int y) {
        rect.y = std::max(0, std::min(y - PADDLE_HEIGHT / 2, GAME_HEIGHT - PADDLE_HEIGHT));
    }
};

struct Ball {
    cv::Point2f position;
    cv::Point2f velocity;
    int radius;
    bool isSuperShot;
    cv::Scalar trailColor;

    Ball() : position(GAME_WIDTH / 2, GAME_HEIGHT / 2),
        velocity(BALL_SPEED, BALL_SPEED), radius(BALL_SIZE / 2),
        isSuperShot(false), trailColor(255, 255, 255) {
        reset();
    }

    void update(float speedMultiplier = 1.0f) {
        position += velocity * speedMultiplier;
        if (position.y <= radius || position.y >= GAME_HEIGHT - radius) {
            velocity.y = -velocity.y;
        }
    }

    void reset() {
        position = cv::Point2f(GAME_WIDTH / 2, GAME_HEIGHT / 2);
        float angle = (rand() % 60 - 30) * CV_PI / 180.0f;
        velocity.x = BALL_SPEED * cos(angle) * (rand() % 2 ? 1 : -1);
        velocity.y = BALL_SPEED * sin(angle);
        isSuperShot = false;
        trailColor = cv::Scalar(255, 255, 255);
    }
};

class ColorPongGesture {
private:
    Paddle player1;
    Paddle player2;
    Ball ball;

    bool gameRunning;
    std::string winner;

    // 트래커와 분석기
    std::unique_ptr<KalmanTracker> player1Tracker;
    std::unique_ptr<KalmanTracker> player2Tracker;
    std::unique_ptr<TrajectoryAnalyzer> player1Analyzer;
    std::unique_ptr<TrajectoryAnalyzer> player2Analyzer;

    // 파워업
    std::vector<PowerUp> player1PowerUps;
    std::vector<PowerUp> player2PowerUps;

    // 파티클
    std::vector<Particle> particles;

    // 공 궤적
    std::deque<cv::Point2f> ballTrail;

public:
    int p1_h_min, p1_h_max, p1_s_min, p1_s_max, p1_v_min, p1_v_max;
    int p2_h_min, p2_h_max, p2_s_min, p2_s_max, p2_v_min, p2_v_max;

    cv::Mat player1_mask;
    cv::Mat player2_mask;

    ColorPongGesture()
        : player1(50, GAME_HEIGHT / 2 - PADDLE_HEIGHT / 2, cv::Scalar(0, 0, 255)),
        player2(GAME_WIDTH - 50 - PADDLE_WIDTH, GAME_HEIGHT / 2 - PADDLE_HEIGHT / 2, cv::Scalar(0, 255, 255)),
        gameRunning(true) {

        player1Tracker = std::make_unique<KalmanTracker>();
        player2Tracker = std::make_unique<KalmanTracker>();
        player1Analyzer = std::make_unique<TrajectoryAnalyzer>();
        player2Analyzer = std::make_unique<TrajectoryAnalyzer>();

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

        // Player 1 추적
        cv::Point2f player1_raw = detectColor(hsv, player1_lower, player1_upper, player1_mask, true);
        bool p1_found = (player1_raw.x > 0);
        cv::Point2f player1_filtered = player1Tracker->update(player1_raw, p1_found);

        if (player1Tracker->isTracking()) {
            player1.updatePosition(player1_filtered.y);
            player1Analyzer->addPoint(player1_filtered);

            // 제스처 감지
            GestureType gesture = player1Analyzer->detectGesture();
            if (gesture != GestureType::NONE) {
                activatePowerUp(1, gesture);
                player1Analyzer->clear();
            }
        }

        // Player 2 추적
        cv::Point2f player2_raw = detectColor(hsv, player2_lower, player2_upper, player2_mask, false);
        bool p2_found = (player2_raw.x > 0);
        cv::Point2f player2_filtered = player2Tracker->update(player2_raw, p2_found);

        if (player2Tracker->isTracking()) {
            player2.updatePosition(player2_filtered.y);
            player2Analyzer->addPoint(player2_filtered);

            // 제스처 감지
            GestureType gesture = player2Analyzer->detectGesture();
            if (gesture != GestureType::NONE) {
                activatePowerUp(2, gesture);
                player2Analyzer->clear();
            }
        }
    }

    void activatePowerUp(int player, GestureType gesture) {
        std::vector<PowerUp>& powerUps = (player == 1) ? player1PowerUps : player2PowerUps;

        switch (gesture) {
        case GestureType::CIRCLE:
            powerUps.emplace_back(gesture, 5.0f, "SUPER SHOT!", cv::Scalar(255, 0, 255));
            createParticleEffect(player == 1 ? player1.rect : player2.rect, cv::Scalar(255, 0, 255));
            break;

        case GestureType::ZIGZAG:
            powerUps.emplace_back(gesture, 3.0f, "SHIELD!", cv::Scalar(0, 255, 255));
            if (player == 1) player1.hasShield = true;
            else player2.hasShield = true;
            break;

        case GestureType::VERTICAL:
            powerUps.emplace_back(gesture, 4.0f, "SPEED UP!", cv::Scalar(255, 255, 0));
            break;

        case GestureType::HORIZONTAL:
            powerUps.emplace_back(gesture, 3.0f, "SLOW BALL!", cv::Scalar(0, 255, 0));
            break;
        }
    }

    void createParticleEffect(const cv::Rect& rect, cv::Scalar color) {
        cv::Point2f center(rect.x + rect.width / 2, rect.y + rect.height / 2);
        for (int i = 0; i < 20; i++) {
            float angle = (i / 20.0f) * 2 * CV_PI;
            cv::Point2f vel(cos(angle) * 5, sin(angle) * 5);
            particles.emplace_back(center, vel, color);
        }
    }

    cv::Point detectColor(const cv::Mat& hsv, const cv::Scalar& lower, const cv::Scalar& upper, cv::Mat& out_mask, bool is_red) {
        // ... (이전과 동일)
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

        // 파워업 업데이트
        updatePowerUps(player1PowerUps);
        updatePowerUps(player2PowerUps);

        // 속도 조절 파워업 확인
        float speedMultiplier = 1.0f;
        for (const auto& power : player1PowerUps) {
            if (power.type == GestureType::HORIZONTAL && power.isActive()) speedMultiplier = 0.5f;
            if (power.type == GestureType::VERTICAL && power.isActive()) speedMultiplier = 1.5f;
        }
        for (const auto& power : player2PowerUps) {
            if (power.type == GestureType::HORIZONTAL && power.isActive()) speedMultiplier = 0.5f;
            if (power.type == GestureType::VERTICAL && power.isActive()) speedMultiplier = 1.5f;
        }

        ball.update(speedMultiplier);

        // 공 궤적 업데이트
        ballTrail.push_back(ball.position);
        if (ballTrail.size() > 20) ballTrail.pop_front();

        // 파티클 업데이트
        particles.erase(
            std::remove_if(particles.begin(), particles.end(),
                [](const Particle& p) { return p.isDead(); }),
            particles.end()
        );
        for (auto& p : particles) {
            p.update();
        }

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

    void updatePowerUps(std::vector<PowerUp>& powerUps) {
        powerUps.erase(
            std::remove_if(powerUps.begin(), powerUps.end(),
                [](const PowerUp& p) { return !p.isActive(); }),
            powerUps.end()
        );

        for (auto& power : powerUps) {
            power.update(0.016f); // 60 FPS

            // Shield 해제
            if (power.type == GestureType::ZIGZAG && !power.isActive()) {
                player1.hasShield = false;
                player2.hasShield = false;
            }
        }
    }

    void checkPaddleCollision(Paddle& paddle) {
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

            bool isPlayer1 = paddle.rect.x < GAME_WIDTH / 2;

            if ((isPlayer1 && ball.velocity.x < 0) || (!isPlayer1 && ball.velocity.x > 0)) {

                // Shield 효과
                if (paddle.hasShield) {
                    ball.velocity.x = -ball.velocity.x * 1.5f;
                    createParticleEffect(paddle.rect, cv::Scalar(0, 255, 255));
                    paddle.hasShield = false;
                }
                else {
                    ball.velocity.x = -ball.velocity.x;
                }

                // Super Shot 확인
                bool hasSuperShot = false;
                if (isPlayer1) {
                    for (const auto& power : player1PowerUps) {
                        if (power.type == GestureType::CIRCLE && power.isActive()) {
                            hasSuperShot = true;
                            break;
                        }
                    }
                }
                else {
                    for (const auto& power : player2PowerUps) {
                        if (power.type == GestureType::CIRCLE && power.isActive()) {
                            hasSuperShot = true;
                            break;
                        }
                    }
                }

                if (hasSuperShot) {
                    ball.velocity.x *= 2.0f;
                    ball.isSuperShot = true;
                    ball.trailColor = cv::Scalar(255, 0, 255);
                    createParticleEffect(paddle.rect, cv::Scalar(255, 0, 255));
                }
                else {
                    ball.isSuperShot = false;
                    ball.trailColor = cv::Scalar(255, 255, 255);
                }

                // 각도 조절
                float paddleCenter = paddleTop + paddle.rect.height / 2.0f;
                float hitPosition = (ball.position.y - paddleCenter) / (paddle.rect.height / 2.0f);
                ball.velocity.y = hitPosition * BALL_SPEED * 0.75f;

                // 위치 조정
                if (isPlayer1) {
                    ball.position.x = paddleRight + ball.radius;
                }
                else {
                    ball.position.x = paddleLeft - ball.radius;
                }
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

        // 제스처 궤적 그리기
        drawTrajectory(frame, player1Analyzer->getPoints(), cv::Scalar(100, 0, 0), 2);
        drawTrajectory(frame, player2Analyzer->getPoints(), cv::Scalar(100, 100, 0), 2);

        // 공 궤적
        for (size_t i = 1; i < ballTrail.size(); i++) {
            int alpha = 255 * i / ballTrail.size();
            cv::Scalar color = ball.isSuperShot ?
                cv::Scalar(alpha, 0, alpha) : cv::Scalar(alpha, alpha, alpha);
            cv::line(frame, ballTrail[i - 1], ballTrail[i], color, 2);
        }

        // 파티클
        for (const auto& p : particles) {
            int alpha = 255 * p.life;
            cv::circle(frame, p.position, 3,
                cv::Scalar(p.color[0] * p.life, p.color[1] * p.life, p.color[2] * p.life), -1);
        }

        // 중앙선
        for (int y = 0; y < GAME_HEIGHT; y += 20) {
            cv::line(frame, cv::Point(GAME_WIDTH / 2, y), cv::Point(GAME_WIDTH / 2, y + 10),
                cv::Scalar(100, 100, 100), 2);
        }

        // 패들
        cv::rectangle(frame, player1.rect, player1.color, -1);
        cv::rectangle(frame, player2.rect, player2.color, -1);

        // Shield 효과
        if (player1.hasShield) {
            cv::rectangle(frame,
                cv::Rect(player1.rect.x - 5, player1.rect.y - 5,
                    player1.rect.width + 10, player1.rect.height + 10),
                cv::Scalar(0, 255, 255), 2);
        }
        if (player2.hasShield) {
            cv::rectangle(frame,
                cv::Rect(player2.rect.x - 5, player2.rect.y - 5,
                    player2.rect.width + 10, player2.rect.height + 10),
                cv::Scalar(0, 255, 255), 2);
        }

        // 공
        cv::circle(frame, cv::Point(ball.position), ball.radius,
            ball.isSuperShot ? cv::Scalar(255, 0, 255) : cv::Scalar(255, 255, 255), -1);

        // 점수
        cv::putText(frame, std::to_string(player1.score), cv::Point(GAME_WIDTH / 2 - 100, 80),
            cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 255, 255), 3);
        cv::putText(frame, std::to_string(player2.score), cv::Point(GAME_WIDTH / 2 + 50, 80),
            cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 255, 255), 3);

        // 파워업 표시
        drawPowerUps(frame, player1PowerUps, 10, 100);
        drawPowerUps(frame, player2PowerUps, GAME_WIDTH - 200, 100);

        // 제스처 가이드
        cv::putText(frame, "Gestures: O=SuperShot, Z=Shield, |=SpeedUp, -=SlowBall",
            cv::Point(10, GAME_HEIGHT - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5,
            cv::Scalar(150, 150, 150), 1);

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

    void drawTrajectory(cv::Mat& frame, const std::deque<cv::Point2f>& points, cv::Scalar color, int thickness) {
        for (size_t i = 1; i < points.size(); i++) {
            int alpha = 255 * i / points.size();
            cv::line(frame, points[i - 1], points[i],
                cv::Scalar(color[0] * alpha / 255, color[1] * alpha / 255, color[2] * alpha / 255),
                thickness);
        }
    }

    void drawPowerUps(cv::Mat& frame, const std::vector<PowerUp>& powerUps, int x, int y) {
        int offset = 0;
        for (const auto& power : powerUps) {
            // 파워업 바
            int barWidth = 180 * power.getProgress();
            cv::rectangle(frame, cv::Point(x, y + offset),
                cv::Point(x + barWidth, y + offset + 20),
                power.color, -1);
            cv::rectangle(frame, cv::Point(x, y + offset),
                cv::Point(x + 180, y + offset + 20),
                cv::Scalar(100, 100, 100), 1);

            // 파워업 이름
            cv::putText(frame, power.name, cv::Point(x + 5, y + offset + 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

            offset += 25;
        }
    }

    void restart() {
        player1.score = 0;
        player2.score = 0;
        player1.hasShield = false;
        player2.hasShield = false;
        ball.reset();
        gameRunning = true;
        winner.clear();

        player1Tracker = std::make_unique<KalmanTracker>();
        player2Tracker = std::make_unique<KalmanTracker>();
        player1Analyzer = std::make_unique<TrajectoryAnalyzer>();
        player2Analyzer = std::make_unique<TrajectoryAnalyzer>();

        player1PowerUps.clear();
        player2PowerUps.clear();
        particles.clear();
        ballTrail.clear();
    }

    bool isRunning() const { return gameRunning; }
};

int main() {
    std::cout << "=== Gesture Recognition Pong Game ===" << std::endl;
    std::cout << "제스처로 특수 능력을 발동하세요!" << std::endl;
    std::cout << "원 그리기: 슈퍼샷 (공 속도 2배)" << std::endl;
    std::cout << "지그재그: 방어막 (공을 더 강하게 튕김)" << std::endl;
    std::cout << "수직선: 공 속도 증가" << std::endl;
    std::cout << "수평선: 공 속도 감소" << std::endl;
    std::cout << "SPACE: Restart, ESC: Exit" << std::endl << std::endl;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Could not open webcam!" << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, GAME_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, GAME_HEIGHT);

    ColorPongGesture game;

    cv::namedWindow("Gesture Pong", cv::WINDOW_NORMAL);
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

        cv::imshow("Gesture Pong", gameFrame);
        cv::imshow("Camera View", frame);
        cv::imshow("Red Mask", game.player1_mask);
        cv::imshow("Yellow Mask", game.player2_mask);

        char key = cv::waitKey(16);
        if (key == 27) break;
        else if (key == ' ' && !game.isRunning()) {
            game.restart();
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}