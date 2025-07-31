#ifndef __OSC_H__
#define __OSC_H__

#include "mainheader.h"

// OSCPack 라이브러리 포함
#include "C:/Libraries/oscpack/osc/OscOutboundPacketStream.h"
#include "C:/Libraries/oscpack/ip/UdpSocket.h"


// 상수 정의
const int OUTPUT_BUFFER_SIZE = 4096;
const int CAMERA_WIDTH = 1280;
const int CAMERA_HEIGHT = 720;


// ===== OSC 송신자 클래스 =====
class OSCSender {

public:
    UdpTransmitSocket* transmitSocket;

    char buffer[OUTPUT_BUFFER_SIZE];

    // 생성자 - OSC 서버 연결 설정
    OSCSender(const std::string& address = "127.0.0.1", int port = 8000);

    // 소멸자
    ~OSCSender();

    // 모든 트래킹 데이터를 한 번에 전송
    void sendAllTrackingData(
        float mouseX, float mouseY, bool mouseClick,
        float p1X, float p1Y, bool p1Tracking,
        float p2X, float p2Y, bool p2Tracking);

    // 개별 데이터 전송 메서드들 (선택적 사용)
    void sendMouseData(float x, float y, bool click);

    void sendPlayerData(int playerNum, float x, float y, bool tracking);

    // 제스처 전송
    void sendGesture(const std::string& gesture);
};



#endif // !__OSC_H__
