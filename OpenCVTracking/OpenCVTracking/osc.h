#ifndef __OSC_H__
#define __OSC_H__

#include "mainheader.h"

// OSCPack ���̺귯�� ����
#include "C:/Libraries/oscpack/osc/OscOutboundPacketStream.h"
#include "C:/Libraries/oscpack/ip/UdpSocket.h"


// ��� ����
const int OUTPUT_BUFFER_SIZE = 4096;
const int CAMERA_WIDTH = 1280;
const int CAMERA_HEIGHT = 720;


// ===== OSC �۽��� Ŭ���� =====
class OSCSender {

public:
    UdpTransmitSocket* transmitSocket;

    char buffer[OUTPUT_BUFFER_SIZE];

    // ������ - OSC ���� ���� ����
    OSCSender(const std::string& address = "127.0.0.1", int port = 8000);

    // �Ҹ���
    ~OSCSender();

    // ��� Ʈ��ŷ �����͸� �� ���� ����
    void sendAllTrackingData(
        float mouseX, float mouseY, bool mouseClick,
        float p1X, float p1Y, bool p1Tracking,
        float p2X, float p2Y, bool p2Tracking);

    // ���� ������ ���� �޼���� (������ ���)
    void sendMouseData(float x, float y, bool click);

    void sendPlayerData(int playerNum, float x, float y, bool tracking);

    // ����ó ����
    void sendGesture(const std::string& gesture);
};



#endif // !__OSC_H__
