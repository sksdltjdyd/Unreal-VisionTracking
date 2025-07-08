// color_pong_ultimate.cpp
// ���� ���콺, ���� �����, �޴� �ý����� ���Ե� �ñ��� Pong ����

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <memory>
#include <deque>
#include <algorithm>
#include <fstream>
#include <ctime>

const int GAME_WIDTH = 1280;
const int GAME_HEIGHT = 720;
const int PADDLE_WIDTH = 20;
const int PADDLE_HEIGHT = 100;
const int BALL_SIZE = 20;
const float BALL_SPEED = 5.0f;
const int WINNING_SCORE = 5;

// ���� ����
enum class GameState {
    MENU,
    PLAYING,
    PAUSED,
    GAME_OVER,
    SETTINGS,
    DRAWING_BOARD,
    REPLAY
};

// ���� ���
enum class GameMode {
    CLASSIC,
    POWER_UP,
    SURVIVAL,
    TIME_ATTACK
};

// ���� ���콺 ����
struct VirtualMouse {
    cv::Point2f position;
    bool isClicking;
    bool wasClicking;
    std::chrono::steady_clock::time_point lastClickTime;
    
    VirtualMouse() : position(GAME_WIDTH/2, GAME_HEIGHT/2), 
                     isClicking(false), wasClicking(false) {}
    
    bool justClicked() {
        return isClicking && !wasClicking;
    }
    
    bool justReleased() {
        return !isClicking && wasClicking;
    }
};

// UI ��ư
struct Button {
    cv::Rect rect;
    std::string text;
    cv::Scalar color;
    cv::Scalar hoverColor;
    bool isHovered;
    
    Button(int x, int y, int w, int h, const std::string& t, cv::Scalar c) 
        : rect(x, y, w, h), text(t), color(c), hoverColor(c * 1.5), isHovered(false) {}
    
    bool contains(const cv::Point2f& pt) {
        return rect.contains(cv::Point(pt));
    }
    
    void draw(cv::Mat& frame) {
        cv::Scalar currentColor = isHovered ? hoverColor : color;
        cv::rectangle(frame, rect, currentColor, -1);
        cv::rectangle(frame, rect, cv::Scalar(255, 255, 255), 2);
        
        cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, nullptr);
        cv::Point textPos(rect.x + (rect.width - textSize.width) / 2,
                         rect.y + (rect.height + textSize.height) / 2);
        cv::putText(frame, text, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.8, 
                   cv::Scalar(255, 255, 255), 2);
    }
};

// ����� ����
class DrawingBoard {
private:
    cv::Mat canvas;
    std::deque<cv::Point> currentStroke;
    std::vector<std::vector<cv::Point>> strokes;
    cv::Scalar currentColor;
    int brushSize;

public:
    DrawingBoard() : brushSize(3), currentColor(255, 255, 255) {
        canvas = cv::Mat::zeros(GAME_HEIGHT, GAME_WIDTH, CV_8UC3);
    }

    void startStroke(const cv::Point& pt) {
        currentStroke.clear();
        currentStroke.push_back(pt);
    }

    void addPoint(const cv::Point& pt) {
        if (!currentStroke.empty()) {
            cv::line(canvas, currentStroke.back(), pt, currentColor, brushSize);
            currentStroke.push_back(pt);
        }
    }

    void endStroke() {
        if (!currentStroke.empty()) {
            strokes.push_back(std::vector<cv::Point>(currentStroke.begin(), currentStroke.end()));
        }
        currentStroke.clear();
    }

    void clear() {
        canvas = cv::Mat::zeros(GAME_HEIGHT, GAME_WIDTH, CV_8UC3);
        strokes.clear();
        currentStroke.clear();
    }

    void setColor(const cv::Scalar& color) { currentColor = color; }
    void setBrushSize(int size) { brushSize = size; }

    // Getter �Լ��� �߰�
    cv::Mat getCanvas() const { return canvas; }
    cv::Scalar getCurrentColor() const { return currentColor; }  // �� �Լ� �߰�!
    int getBrushSize() const { return brushSize; }  // �̰͵� �߰�
};

// ���÷��� �ý���
struct GameSnapshot {
    cv::Point2f ballPos;
    cv::Point2f ballVel;
    int paddle1Y;
    int paddle2Y;
    int score1;
    int score2;
    double timestamp;
};

class ReplaySystem {
private:
    std::vector<GameSnapshot> snapshots;
    size_t currentFrame;
    bool isRecording;
    bool isPlaying;
    
public:
    ReplaySystem() : currentFrame(0), isRecording(false), isPlaying(false) {}
    
    void startRecording() {
        snapshots.clear();
        isRecording = true;
        isPlaying = false;
    }
    
    void stopRecording() {
        isRecording = false;
    }
    
    void addSnapshot(const GameSnapshot& snapshot) {
        if (isRecording) {
            snapshots.push_back(snapshot);
            // �ִ� 30�� (60fps * 30)
            if (snapshots.size() > 1800) {
                snapshots.erase(snapshots.begin());
            }
        }
    }
    
    void startPlayback() {
        if (!snapshots.empty()) {
            currentFrame = 0;
            isPlaying = true;
            isRecording = false;
        }
    }
    
    GameSnapshot getNextFrame() {
        if (isPlaying && currentFrame < snapshots.size()) {
            return snapshots[currentFrame++];
        }
        return GameSnapshot();
    }
    
    bool hasMoreFrames() const {
        return isPlaying && currentFrame < snapshots.size();
    }
    
    void stopPlayback() {
        isPlaying = false;
    }
    
    size_t getFrameCount() const { return snapshots.size(); }
    size_t getCurrentFrame() const { return currentFrame; }
};

// ����
struct GameSettings {
    int difficulty;  // 1-5
    float soundVolume;  // 0-1
    bool showEffects;
    bool showTrajectory;
    GameMode mode;
    
    GameSettings() : difficulty(3), soundVolume(0.5f), 
                    showEffects(true), showTrajectory(true), 
                    mode(GameMode::CLASSIC) {}
};

// ���� Ŭ������ (����ȭ)
class KalmanTracker {
    // ... (������ ����)
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
            0, 0, 1, 0,
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
        } else {
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

// ���� ���� Ŭ����
class ColorPongUltimate {
private:
    // ���� ����
    GameState currentState;
    GameMode gameMode;
    GameSettings settings;
    
    // ���� ��ü��
    struct Paddle {
        cv::Rect rect;
        cv::Scalar color;
        int score;
        
        Paddle(int x, int y, cv::Scalar c) 
            : rect(x, y, PADDLE_WIDTH, PADDLE_HEIGHT), color(c), score(0) {}
        
        void updatePosition(int y) {
            rect.y = std::max(0, std::min(y - PADDLE_HEIGHT/2, GAME_HEIGHT - PADDLE_HEIGHT));
        }
    } player1, player2;
    
    struct Ball {
        cv::Point2f position;
        cv::Point2f velocity;
        int radius;
        
        Ball() : position(GAME_WIDTH/2, GAME_HEIGHT/2), 
                 velocity(BALL_SPEED, BALL_SPEED), radius(BALL_SIZE/2) {
            reset();
        }
        
        void update(float speedMultiplier = 1.0f) {
            position += velocity * speedMultiplier;
            if (position.y <= radius || position.y >= GAME_HEIGHT - radius) {
                velocity.y = -velocity.y;
            }
        }
        
        void reset() {
            position = cv::Point2f(GAME_WIDTH/2, GAME_HEIGHT/2);
            float angle = (rand() % 60 - 30) * CV_PI / 180.0f;
            velocity.x = BALL_SPEED * cos(angle) * (rand() % 2 ? 1 : -1);
            velocity.y = BALL_SPEED * sin(angle);
        }
    } ball;
    
    // ���� �ý���
    std::unique_ptr<KalmanTracker> mouseTracker;
    std::unique_ptr<KalmanTracker> player1Tracker;
    std::unique_ptr<KalmanTracker> player2Tracker;
    
    // UI �ý���
    VirtualMouse virtualMouse;
    std::vector<Button> menuButtons;
    std::vector<Button> settingsButtons;
    std::vector<Button> pauseButtons;
    
    // ����� & ���÷���
    DrawingBoard drawingBoard;
    ReplaySystem replaySystem;
    
    // ���� ���
    struct GameStats {
        int totalHits;
        int longestRally;
        int currentRally;
        float averageRallyLength;
        std::vector<float> rallyHistory;
        
        void reset() {
            totalHits = 0;
            longestRally = 0;
            currentRally = 0;
            averageRallyLength = 0;
            rallyHistory.clear();
        }
    } stats;
    
public:
    // HSV ����
    int mouse_h_min, mouse_h_max, mouse_s_min, mouse_s_max, mouse_v_min, mouse_v_max;
    int p1_h_min, p1_h_max, p1_s_min, p1_s_max, p1_v_min, p1_v_max;
    int p2_h_min, p2_h_max, p2_s_min, p2_s_max, p2_v_min, p2_v_max;
    
    cv::Mat mouse_mask, player1_mask, player2_mask;

    // Getter �Լ� �߰�
    GameState getCurrentState() const { return currentState; }

    ColorPongUltimate() 
        : currentState(GameState::MENU),
          player1(50, GAME_HEIGHT/2 - PADDLE_HEIGHT/2, cv::Scalar(0, 0, 255)),
          player2(GAME_WIDTH - 50 - PADDLE_WIDTH, GAME_HEIGHT/2 - PADDLE_HEIGHT/2, cv::Scalar(0, 255, 255)) {
        
        // Ʈ��Ŀ �ʱ�ȭ
        mouseTracker = std::make_unique<KalmanTracker>();
        player1Tracker = std::make_unique<KalmanTracker>();
        player2Tracker = std::make_unique<KalmanTracker>();
        
        // ���콺�� �ʷϻ� HSV �⺻��
        mouse_h_min = 45;  mouse_h_max = 85;
        mouse_s_min = 0; mouse_s_max = 249;
        mouse_v_min = 0; mouse_v_max = 73;
        
        // Player HSV �⺻��
        p1_h_min = 0;   p1_h_max = 5;
        p1_s_min = 0; p1_s_max = 255;
        p1_v_min = 61; p1_v_max = 110;
        
        p2_h_min = 20;  p2_h_max = 25;
        p2_s_min = 100; p2_s_max = 255;
        p2_v_min = 100; p2_v_max = 255;
        
        initializeUI();
    }
    
    void initializeUI() {
        // ���� �޴� ��ư
        menuButtons.clear();
        menuButtons.emplace_back(GAME_WIDTH/2 - 150, 200, 300, 60, "CLASSIC MODE", cv::Scalar(100, 100, 100));
        menuButtons.emplace_back(GAME_WIDTH/2 - 150, 280, 300, 60, "POWER-UP MODE", cv::Scalar(100, 100, 100));
        menuButtons.emplace_back(GAME_WIDTH/2 - 150, 360, 300, 60, "DRAWING BOARD", cv::Scalar(100, 100, 100));
        menuButtons.emplace_back(GAME_WIDTH/2 - 150, 440, 300, 60, "SETTINGS", cv::Scalar(100, 100, 100));
        menuButtons.emplace_back(GAME_WIDTH/2 - 150, 520, 300, 60, "QUIT", cv::Scalar(100, 100, 100));
        
        // �Ͻ����� �޴� ��ư
        pauseButtons.clear();
        pauseButtons.emplace_back(GAME_WIDTH/2 - 150, 250, 300, 60, "RESUME", cv::Scalar(100, 100, 100));
        pauseButtons.emplace_back(GAME_WIDTH/2 - 150, 330, 300, 60, "MAIN MENU", cv::Scalar(100, 100, 100));
        pauseButtons.emplace_back(GAME_WIDTH/2 - 150, 410, 300, 60, "VIEW REPLAY", cv::Scalar(100, 100, 100));
        
        // ���� ��ư
        settingsButtons.clear();
        settingsButtons.emplace_back(GAME_WIDTH/2 - 150, 200, 300, 60, "DIFFICULTY: " + std::to_string(settings.difficulty), cv::Scalar(100, 100, 100));
        settingsButtons.emplace_back(GAME_WIDTH/2 - 150, 280, 300, 60, settings.showEffects ? "EFFECTS: ON" : "EFFECTS: OFF", cv::Scalar(100, 100, 100));
        settingsButtons.emplace_back(GAME_WIDTH/2 - 150, 360, 300, 60, settings.showTrajectory ? "TRAJECTORY: ON" : "TRAJECTORY: OFF", cv::Scalar(100, 100, 100));
        settingsButtons.emplace_back(GAME_WIDTH/2 - 150, 440, 300, 60, "BACK", cv::Scalar(100, 100, 100));
    }
    
    void updateTracking(const cv::Mat& frame) {
        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        // ���� ���콺 ���� (�ʷϻ�)
        cv::Scalar mouse_lower(mouse_h_min, mouse_s_min, mouse_v_min);
        cv::Scalar mouse_upper(mouse_h_max, mouse_s_max, mouse_v_max);
        cv::Point2f mouse_raw = detectColor(hsv, mouse_lower, mouse_upper, mouse_mask);
        bool mouse_found = (mouse_raw.x > 0);
        cv::Point2f mouse_filtered = mouseTracker->update(mouse_raw, mouse_found);
        
        if (mouseTracker->isTracking()) {
            virtualMouse.position = mouse_filtered;
            
            // Ŭ�� ���� (�ָ� ��� = ���� ����)
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mouse_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            
            if (!contours.empty()) {
                double maxArea = 0;
                for (const auto& contour : contours) {
                    maxArea = std::max(maxArea, cv::contourArea(contour));
                }
                
                // ������ �پ��� Ŭ������ �ν�
                virtualMouse.wasClicking = virtualMouse.isClicking;
                virtualMouse.isClicking = (maxArea < 2000);  // �Ӱ谪 ���� �ʿ�
            }
        }
        
        // ���� ���� ���� �÷��̾� ����
        if (currentState == GameState::PLAYING) {
            // Player 1 ����
            cv::Scalar player1_lower(p1_h_min, p1_s_min, p1_v_min);
            cv::Scalar player1_upper(p1_h_max, p1_s_max, p1_v_max);
            cv::Point2f player1_raw = detectColor(hsv, player1_lower, player1_upper, player1_mask);
            bool p1_found = (player1_raw.x > 0);
            cv::Point2f player1_filtered = player1Tracker->update(player1_raw, p1_found);
            
            if (player1Tracker->isTracking()) {
                player1.updatePosition(player1_filtered.y);
            }
            
            // Player 2 ����
            cv::Scalar player2_lower(p2_h_min, p2_s_min, p2_v_min);
            cv::Scalar player2_upper(p2_h_max, p2_s_max, p2_v_max);
            cv::Point2f player2_raw = detectColor(hsv, player2_lower, player2_upper, player2_mask);
            bool p2_found = (player2_raw.x > 0);
            cv::Point2f player2_filtered = player2Tracker->update(player2_raw, p2_found);
            
            if (player2Tracker->isTracking()) {
                player2.updatePosition(player2_filtered.y);
            }
        }
    }
    
    cv::Point detectColor(const cv::Mat& hsv, const cv::Scalar& lower, const cv::Scalar& upper, cv::Mat& out_mask) {
        // out_mask �ʱ�ȭ Ȯ��
        if (out_mask.empty() || out_mask.size() != hsv.size()) {
            out_mask = cv::Mat::zeros(hsv.size(), CV_8UC1);
        }

        cv::inRange(hsv, lower, upper, out_mask);

        cv::erode(out_mask, out_mask, cv::Mat(), cv::Point(-1, -1), 2);
        cv::dilate(out_mask, out_mask, cv::Mat(), cv::Point(-1, -1), 2);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(out_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if (!contours.empty()) {
            double maxArea = 0;
            int maxIdx = -1;

            for (int i = 0; i < contours.size(); i++) {
                double area = cv::contourArea(contours[i]);
                if (area > maxArea && area > 500) {
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
    
    void update() {
        switch (currentState) {
            case GameState::MENU:
                updateMenu();
                break;
                
            case GameState::PLAYING:
                updateGame();
                break;
                
            case GameState::PAUSED:
                updatePause();
                break;
                
            case GameState::SETTINGS:
                updateSettings();
                break;
                
            case GameState::DRAWING_BOARD:
                updateDrawing();
                break;
                
            case GameState::REPLAY:
                updateReplay();
                break;
                
            case GameState::GAME_OVER:
                updateGameOver();
                break;
        }
    }
    
    void updateMenu() {
        // ��ư ȣ�� üũ
        for (auto& btn : menuButtons) {
            btn.isHovered = btn.contains(virtualMouse.position);
        }
        
        // Ŭ�� üũ
        if (virtualMouse.justClicked()) {
            if (menuButtons[0].contains(virtualMouse.position)) {
                startGame(GameMode::CLASSIC);
            } else if (menuButtons[1].contains(virtualMouse.position)) {
                startGame(GameMode::POWER_UP);
            } else if (menuButtons[2].contains(virtualMouse.position)) {
                currentState = GameState::DRAWING_BOARD;
                drawingBoard.clear();
            } else if (menuButtons[3].contains(virtualMouse.position)) {
                currentState = GameState::SETTINGS;
            } else if (menuButtons[4].contains(virtualMouse.position)) {
                // ����
                exit(0);
            }
        }
    }
    
    void updateGame() {
        // �Ͻ����� üũ (PŰ �Ǵ� ����ó)
        if (checkPauseGesture()) {
            currentState = GameState::PAUSED;
            return;
        }
        
        // ���� ����
        ball.update(settings.difficulty * 0.3f + 0.7f);
        checkPaddleCollision(player1);
        checkPaddleCollision(player2);
        
        // ���� üũ
        if (ball.position.x <= 0) {
            player2.score++;
            stats.rallyHistory.push_back(stats.currentRally);
            stats.currentRally = 0;
            ball.reset();
            checkWinner();
        } else if (ball.position.x >= GAME_WIDTH) {
            player1.score++;
            stats.rallyHistory.push_back(stats.currentRally);
            stats.currentRally = 0;
            ball.reset();
            checkWinner();
        }
        
        // ���÷��� ���
        if (replaySystem.getFrameCount() == 0) {
            replaySystem.startRecording();
        }
        
        GameSnapshot snapshot;
        snapshot.ballPos = ball.position;
        snapshot.ballVel = ball.velocity;
        snapshot.paddle1Y = player1.rect.y;
        snapshot.paddle2Y = player2.rect.y;
        snapshot.score1 = player1.score;
        snapshot.score2 = player2.score;
        snapshot.timestamp = cv::getTickCount() / cv::getTickFrequency();
        replaySystem.addSnapshot(snapshot);
    }
    
    void updatePause() {
        for (auto& btn : pauseButtons) {
            btn.isHovered = btn.contains(virtualMouse.position);
        }
        
        if (virtualMouse.justClicked()) {
            if (pauseButtons[0].contains(virtualMouse.position)) {
                currentState = GameState::PLAYING;
            } else if (pauseButtons[1].contains(virtualMouse.position)) {
                currentState = GameState::MENU;
                resetGame();
            } else if (pauseButtons[2].contains(virtualMouse.position)) {
                currentState = GameState::REPLAY;
                replaySystem.startPlayback();
            }
        }
    }
    
    void updateSettings() {
        for (auto& btn : settingsButtons) {
            btn.isHovered = btn.contains(virtualMouse.position);
        }
        
        if (virtualMouse.justClicked()) {
            if (settingsButtons[0].contains(virtualMouse.position)) {
                settings.difficulty = (settings.difficulty % 5) + 1;
                settingsButtons[0].text = "DIFFICULTY: " + std::to_string(settings.difficulty);
            } else if (settingsButtons[1].contains(virtualMouse.position)) {
                settings.showEffects = !settings.showEffects;
                settingsButtons[1].text = settings.showEffects ? "EFFECTS: ON" : "EFFECTS: OFF";
            } else if (settingsButtons[2].contains(virtualMouse.position)) {
                settings.showTrajectory = !settings.showTrajectory;
                settingsButtons[2].text = settings.showTrajectory ? "TRAJECTORY: ON" : "TRAJECTORY: OFF";
            } else if (settingsButtons[3].contains(virtualMouse.position)) {
                currentState = GameState::MENU;
                saveSettings();
            }
        }
    }
    
    void updateDrawing() {
        if (virtualMouse.isClicking) {
            if (!virtualMouse.wasClicking) {
                drawingBoard.startStroke(cv::Point(virtualMouse.position));
            } else {
                drawingBoard.addPoint(cv::Point(virtualMouse.position));
            }
        } else if (virtualMouse.wasClicking) {
            drawingBoard.endStroke();
        }
        
        // ���� ���� (���� ����ó)
        if (checkColorChangeGesture()) {
            static int colorIndex = 0;
            cv::Scalar colors[] = {
                cv::Scalar(255, 255, 255),  // ���
                cv::Scalar(0, 0, 255),      // ����
                cv::Scalar(0, 255, 0),      // �ʷ�
                cv::Scalar(255, 0, 0),      // �Ķ�
                cv::Scalar(0, 255, 255)     // ���
            };
            colorIndex = (colorIndex + 1) % 5;
            drawingBoard.setColor(colors[colorIndex]);
        }
    }
    
    void updateReplay() {
        if (replaySystem.hasMoreFrames()) {
            GameSnapshot snapshot = replaySystem.getNextFrame();
            ball.position = snapshot.ballPos;
            ball.velocity = snapshot.ballVel;
            player1.rect.y = snapshot.paddle1Y;
            player2.rect.y = snapshot.paddle2Y;
            player1.score = snapshot.score1;
            player2.score = snapshot.score2;
        } else {
            currentState = GameState::PAUSED;
            replaySystem.stopPlayback();
        }
    }
    
    void updateGameOver() {
        static Button restartButton(GAME_WIDTH/2 - 150, 400, 300, 60, "PLAY AGAIN", cv::Scalar(100, 100, 100));
        static Button menuButton(GAME_WIDTH/2 - 150, 480, 300, 60, "MAIN MENU", cv::Scalar(100, 100, 100));
        
        restartButton.isHovered = restartButton.contains(virtualMouse.position);
        menuButton.isHovered = menuButton.contains(virtualMouse.position);
        
        if (virtualMouse.justClicked()) {
            if (restartButton.contains(virtualMouse.position)) {
                startGame(gameMode);
            } else if (menuButton.contains(virtualMouse.position)) {
                currentState = GameState::MENU;
                resetGame();
            }
        }
    }
    
    void draw(cv::Mat& frame) {
        frame = cv::Scalar(0, 0, 0);
        
        switch (currentState) {
            case GameState::MENU:
                drawMenu(frame);
                break;
                
            case GameState::PLAYING:
            case GameState::REPLAY:
                drawGame(frame);
                break;
                
            case GameState::PAUSED:
                drawGame(frame);
                drawPause(frame);
                break;
                
            case GameState::SETTINGS:
                drawSettings(frame);
                break;
                
            case GameState::DRAWING_BOARD:
                drawDrawingBoard(frame);
                break;
                
            case GameState::GAME_OVER:
                drawGame(frame);
                drawGameOver(frame);
                break;
        }
        
        // ���� ���콺 Ŀ�� �׸���
        drawVirtualMouse(frame);
    }
    
    void drawMenu(cv::Mat& frame) {
        // Ÿ��Ʋ
        cv::putText(frame, "ULTIMATE PONG", cv::Point(GAME_WIDTH/2 - 200, 100), 
                   cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 255, 255), 3);
        
        // ��ư��
        for (auto& btn : menuButtons) {
            btn.draw(frame);
        }
        
        // �ȳ�
        cv::putText(frame, "Use GREEN object as mouse", cv::Point(GAME_WIDTH/2 - 150, 600), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, "Make a fist to click", cv::Point(GAME_WIDTH/2 - 100, 630), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }
    
    void drawGame(cv::Mat& frame) {
        // �߾Ӽ�
        for (int y = 0; y < GAME_HEIGHT; y += 20) {
            cv::line(frame, cv::Point(GAME_WIDTH/2, y), cv::Point(GAME_WIDTH/2, y + 10), 
                    cv::Scalar(100, 100, 100), 2);
        }
        
        // �е�
        cv::rectangle(frame, player1.rect, player1.color, -1);
        cv::rectangle(frame, player2.rect, player2.color, -1);
        
        // ��
        cv::circle(frame, cv::Point(ball.position), ball.radius, cv::Scalar(255, 255, 255), -1);
        
        // ����
        cv::putText(frame, std::to_string(player1.score), cv::Point(GAME_WIDTH/2 - 100, 80), 
                    cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 255, 255), 3);
        cv::putText(frame, std::to_string(player2.score), cv::Point(GAME_WIDTH/2 + 50, 80), 
                    cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 255, 255), 3);
        
        // ���� ���
        std::string statsText = "Rally: " + std::to_string(stats.currentRally) + 
                               " | Best: " + std::to_string(stats.longestRally);
        cv::putText(frame, statsText, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);
        
        // ���÷��� ��� ǥ��
        if (currentState == GameState::REPLAY) {
            cv::putText(frame, "REPLAY MODE", cv::Point(GAME_WIDTH/2 - 100, 150), 
                       cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 0), 2);
            
            // ���� ��
            int barWidth = 400;
            int progress = (replaySystem.getCurrentFrame() * barWidth) / replaySystem.getFrameCount();
            cv::rectangle(frame, cv::Point(GAME_WIDTH/2 - 200, 180), 
                         cv::Point(GAME_WIDTH/2 - 200 + progress, 190), 
                         cv::Scalar(255, 255, 0), -1);
            cv::rectangle(frame, cv::Point(GAME_WIDTH/2 - 200, 180), 
                         cv::Point(GAME_WIDTH/2 + 200, 190), 
                         cv::Scalar(100, 100, 100), 1);
        }
    }
    
    void drawPause(cv::Mat& frame) {
        // ������ ��������
        cv::Mat overlay = frame.clone();
        cv::rectangle(overlay, cv::Point(0, 0), cv::Point(GAME_WIDTH, GAME_HEIGHT), 
                      cv::Scalar(0, 0, 0), -1);
        cv::addWeighted(overlay, 0.7, frame, 0.3, 0, frame);
        
        cv::putText(frame, "PAUSED", cv::Point(GAME_WIDTH/2 - 100, 150), 
                   cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 255, 255), 3);
        
        for (auto& btn : pauseButtons) {
            btn.draw(frame);
        }
    }
    
    void drawSettings(cv::Mat& frame) {
        cv::putText(frame, "SETTINGS", cv::Point(GAME_WIDTH/2 - 100, 100), 
                   cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255, 255, 255), 3);
        
        for (auto& btn : settingsButtons) {
            btn.draw(frame);
        }
    }
    
    void drawDrawingBoard(cv::Mat& frame) {
        // ����� ĵ����
        frame = drawingBoard.getCanvas();

        // UI
        cv::rectangle(frame, cv::Point(0, 0), cv::Point(GAME_WIDTH, 50), cv::Scalar(50, 50, 50), -1);
        cv::putText(frame, "DRAWING BOARD - Draw strategy or messages!", cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

        // ��Ʈ��
        cv::putText(frame, "Vertical gesture: Change color | Horizontal: Clear | Circle: Back to menu",
            cv::Point(10, GAME_HEIGHT - 20),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);

        // ���� �귯�� ǥ�� - getCurrentColor() ���
        cv::circle(frame, cv::Point(GAME_WIDTH - 50, 25), 10, drawingBoard.getCurrentColor(), -1);
    }
    
    void drawGameOver(cv::Mat& frame) {
        cv::Mat overlay = frame.clone();
        cv::rectangle(overlay, cv::Point(0, 0), cv::Point(GAME_WIDTH, GAME_HEIGHT), 
                      cv::Scalar(0, 0, 0), -1);
        cv::addWeighted(overlay, 0.7, frame, 0.3, 0, frame);
        
        std::string winner = (player1.score >= WINNING_SCORE) ? "RED WINS!" : "YELLOW WINS!";
        cv::Size textSize = cv::getTextSize(winner, cv::FONT_HERSHEY_SIMPLEX, 3, 4, nullptr);
        cv::putText(frame, winner, cv::Point((GAME_WIDTH - textSize.width)/2, 200), 
                   cv::FONT_HERSHEY_SIMPLEX, 3, cv::Scalar(255, 255, 255), 4);
        
        // ���
        std::string statsText = "Longest Rally: " + std::to_string(stats.longestRally) + 
                               " | Total Hits: " + std::to_string(stats.totalHits);
        cv::putText(frame, statsText, cv::Point(GAME_WIDTH/2 - 200, 300), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(200, 200, 200), 2);
        
        static Button restartButton(GAME_WIDTH/2 - 150, 400, 300, 60, "PLAY AGAIN", cv::Scalar(100, 100, 100));
        static Button menuButton(GAME_WIDTH/2 - 150, 480, 300, 60, "MAIN MENU", cv::Scalar(100, 100, 100));
        
        restartButton.draw(frame);
        menuButton.draw(frame);
    }
    
    void drawVirtualMouse(cv::Mat& frame) {
        if (mouseTracker->isTracking()) {
            // ���콺 Ŀ��
            cv::circle(frame, cv::Point(virtualMouse.position), 15, 
                      virtualMouse.isClicking ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 200, 0), 2);
            cv::circle(frame, cv::Point(virtualMouse.position), 3, cv::Scalar(0, 255, 0), -1);
            
            // ���콺 ����
            if (settings.showTrajectory) {
                for (size_t i = 1; i < mouseTracker->trajectory.size(); i++) {
                    int alpha = 255 * i / mouseTracker->trajectory.size();
                    cv::line(frame, mouseTracker->trajectory[i-1], mouseTracker->trajectory[i],
                            cv::Scalar(0, alpha, 0), 1);
                }
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
                ball.velocity.x = -ball.velocity.x;
                
                float paddleCenter = paddleTop + paddle.rect.height / 2.0f;
                float hitPosition = (ball.position.y - paddleCenter) / (paddle.rect.height / 2.0f);
                ball.velocity.y = hitPosition * BALL_SPEED * 0.75f;
                
                if (isPlayer1) {
                    ball.position.x = paddleRight + ball.radius;
                } else {
                    ball.position.x = paddleLeft - ball.radius;
                }
                
                // ��� ������Ʈ
                stats.totalHits++;
                stats.currentRally++;
                stats.longestRally = std::max(stats.longestRally, stats.currentRally);
            }
        }
    }
    
    bool checkPauseGesture() {
        // ��� �ø��� ����ó (�� �����Ⱑ ȭ�� ��ܿ� ���� ��)
        if (player1Tracker->isTracking() && player2Tracker->isTracking()) {
            if (player1Tracker->lastPosition.y < 100 && player2Tracker->lastPosition.y < 100) {
                return true;
            }
        }
        return false;
    }
    
    bool checkColorChangeGesture() {
        // ���� ����ó ����
        if (mouseTracker->trajectory.size() > 20) {
            float totalDy = 0;
            float totalDx = 0;
            for (size_t i = 1; i < mouseTracker->trajectory.size(); i++) {
                totalDy += abs(mouseTracker->trajectory[i].y - mouseTracker->trajectory[i-1].y);
                totalDx += abs(mouseTracker->trajectory[i].x - mouseTracker->trajectory[i-1].x);
            }
            return totalDy > totalDx * 3 && totalDy > 100;
        }
        return false;
    }
    
    void startGame(GameMode mode) {
        gameMode = mode;
        currentState = GameState::PLAYING;
        resetGame();
        stats.reset();
        replaySystem.startRecording();
    }
    
    void resetGame() {
        player1.score = 0;
        player2.score = 0;
        ball.reset();
        stats.currentRally = 0;
    }
    
    void checkWinner() {
        if (player1.score >= WINNING_SCORE || player2.score >= WINNING_SCORE) {
            currentState = GameState::GAME_OVER;
            replaySystem.stopRecording();
            
            // ��� ���� ���
            if (!stats.rallyHistory.empty()) {
                float sum = 0;
                for (float rally : stats.rallyHistory) {
                    sum += rally;
                }
                stats.averageRallyLength = sum / stats.rallyHistory.size();
            }
        }
    }
    
    void saveSettings() {
        std::ofstream file("pong_settings.txt");
        if (file.is_open()) {
            file << settings.difficulty << std::endl;
            file << settings.showEffects << std::endl;
            file << settings.showTrajectory << std::endl;
            file.close();
        }
    }
    
    void loadSettings() {
        std::ifstream file("pong_settings.txt");
        if (file.is_open()) {
            file >> settings.difficulty;
            file >> settings.showEffects;
            file >> settings.showTrajectory;
            file.close();
        }
    }
    
    void handleKey(char key) {
        if (key == 'p' || key == 'P') {
            if (currentState == GameState::PLAYING) {
                currentState = GameState::PAUSED;
            } else if (currentState == GameState::PAUSED) {
                currentState = GameState::PLAYING;
            }
        } else if (key == 27) { // ESC
            if (currentState == GameState::DRAWING_BOARD) {
                currentState = GameState::MENU;
            }
        }
    }
};

int main() {
    std::cout << "=== ULTIMATE COLOR PONG ===" << std::endl;
    std::cout << "���� ���콺 �ý������� ��� ���� �����ϼ���!" << std::endl;
    std::cout << "�ʷϻ� ��ü: ���콺 Ŀ��" << std::endl;
    std::cout << "�ָ� ���: Ŭ��" << std::endl;
    std::cout << "������ ��ü: Player 1 (����)" << std::endl;
    std::cout << "����� ��ü: Player 2 (������)" << std::endl;
    std::cout << "��� ���: ���� �Ͻ�����" << std::endl << std::endl;

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Could not open webcam!" << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, GAME_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, GAME_HEIGHT);

    ColorPongUltimate game;
    game.loadSettings();

    // ����ũ �ʱ�ȭ
    game.mouse_mask = cv::Mat::zeros(GAME_HEIGHT, GAME_WIDTH, CV_8UC1);
    game.player1_mask = cv::Mat::zeros(GAME_HEIGHT, GAME_WIDTH, CV_8UC1);
    game.player2_mask = cv::Mat::zeros(GAME_HEIGHT, GAME_WIDTH, CV_8UC1);

    cv::namedWindow("Ultimate Pong", cv::WINDOW_NORMAL);
    cv::namedWindow("Camera View", cv::WINDOW_NORMAL);
    cv::namedWindow("HSV Controls", cv::WINDOW_NORMAL);
    cv::namedWindow("Green Mask (Mouse)", cv::WINDOW_NORMAL);
    cv::namedWindow("Red Mask (P1)", cv::WINDOW_NORMAL);
    cv::namedWindow("Yellow Mask (P2)", cv::WINDOW_NORMAL);

    cv::resizeWindow("Camera View", 480, 270);
    cv::resizeWindow("Green Mask (Mouse)", 320, 180);
    cv::resizeWindow("Red Mask (P1)", 320, 180);
    cv::resizeWindow("Yellow Mask (P2)", 320, 180);

    // HSV Ʈ���� ����
    // ���콺 (�ʷϻ�)
    cv::createTrackbar("Mouse H_MIN", "HSV Controls", &game.mouse_h_min, 180);
    cv::createTrackbar("Mouse H_MAX", "HSV Controls", &game.mouse_h_max, 180);
    cv::createTrackbar("Mouse S_MIN", "HSV Controls", &game.mouse_s_min, 255);
    cv::createTrackbar("Mouse S_MAX", "HSV Controls", &game.mouse_s_max, 255);
    cv::createTrackbar("Mouse V_MIN", "HSV Controls", &game.mouse_v_min, 255);
    cv::createTrackbar("Mouse V_MAX", "HSV Controls", &game.mouse_v_max, 255);

    // Player 1 (����)
    cv::createTrackbar("P1 H_MIN", "HSV Controls", &game.p1_h_min, 180);
    cv::createTrackbar("P1 H_MAX", "HSV Controls", &game.p1_h_max, 180);
    cv::createTrackbar("P1 S_MIN", "HSV Controls", &game.p1_s_min, 255);
    cv::createTrackbar("P1 S_MAX", "HSV Controls", &game.p1_s_max, 255);
    cv::createTrackbar("P1 V_MIN", "HSV Controls", &game.p1_v_min, 255);
    cv::createTrackbar("P1 V_MAX", "HSV Controls", &game.p1_v_max, 255);

    // Player 2 (���)
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
        game.update();
        game.draw(gameFrame);

        cv::imshow("Ultimate Pong", gameFrame);
        cv::imshow("Camera View", frame);

        // ����ũ�� ������� ���� ���� ǥ��
        if (!game.mouse_mask.empty()) {
            cv::imshow("Green Mask (Mouse)", game.mouse_mask);
        }
        if (!game.player1_mask.empty()) {
            cv::imshow("Red Mask (P1)", game.player1_mask);
        }
        if (!game.player2_mask.empty()) {
            cv::imshow("Yellow Mask (P2)", game.player2_mask);
        }

        char key = cv::waitKey(16);
        if (key == 27 && game.getCurrentState() == GameState::MENU) break;
        game.handleKey(key);
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}