
#include "osc.h"

OSCSender::OSCSender(const std::string& address, int port) {
    try {
        // UDP ������ ���� OSC �޽��� ����
        transmitSocket = new UdpTransmitSocket(
            IpEndpointName(address.c_str(), port)
        );
        std::cout << "OSC Sender initialized: " << address << ":" << port << std::endl;
    }
    catch (std::exception& e) {
        std::cerr << "OSC �ʱ�ȭ ����: " << e.what() << std::endl;
        transmitSocket = nullptr;
    }
}

OSCSender::~OSCSender() {
    if (transmitSocket) {
        delete transmitSocket;
    }
}

// ��� Ʈ��ŷ �����͸� �� ���� ����
void OSCSender::sendAllTrackingData(
    float mouseX, float mouseY, bool mouseClick,
    float p1X, float p1Y, bool p1Tracking,
    float p2X, float p2Y, bool p2Tracking) {

    if (!transmitSocket) return;

    // OSC ��Ŷ ��Ʈ�� ����
    osc::OutboundPacketStream packet(buffer, OUTPUT_BUFFER_SIZE);

    try {
        // OSC �޽��� ����
        // �ּ� ����: /tracking/all
        // ������: mouseX, mouseY, mouseClick, p1X, p1Y, p2X, p2Y, p1Track, p2Track
        packet << osc::BeginMessage("/tracking/all")
            << mouseX << mouseY << (mouseClick ? 1 : 0)    // ���콺 ������
            << p1X << p1Y                                  // Player 1 ��ġ
            << p2X << p2Y                                  // Player 2 ��ġ
            << (p1Tracking ? 1 : 0)                        // Player 1 ���� ����
            << (p2Tracking ? 1 : 0)                        // Player 2 ���� ����
            << osc::EndMessage;

        // ��Ŷ ����
        transmitSocket->Send(packet.Data(), packet.Size());

    }
    catch (std::exception& e) {
        std::cerr << "OSC ���� ����: " << e.what() << std::endl;
    }
}


// ���� ������ ���� �޼���� (������ ���)
void OSCSender::sendMouseData(float x, float y, bool click) {
    if (!transmitSocket) return;

    osc::OutboundPacketStream packet(buffer, OUTPUT_BUFFER_SIZE);
    packet << osc::BeginMessage("/tracking/mouse")
        << x << y << (click ? 1 : 0)
        << osc::EndMessage;

    transmitSocket->Send(packet.Data(), packet.Size());
}

void OSCSender::sendPlayerData(int playerNum, float x, float y, bool tracking) {
    if (!transmitSocket) return;

    osc::OutboundPacketStream packet(buffer, OUTPUT_BUFFER_SIZE);
    std::string address = (playerNum == 1) ? "/tracking/player1" : "/tracking/player2";

    packet << osc::BeginMessage(address.c_str())
        << x << y << (tracking ? 1 : 0)
        << osc::EndMessage;

    transmitSocket->Send(packet.Data(), packet.Size());
}

// ����ó ����
void OSCSender::sendGesture(const std::string& gesture) {
    if (!transmitSocket) return;

    osc::OutboundPacketStream packet(buffer, OUTPUT_BUFFER_SIZE);
    packet << osc::BeginMessage("/tracking/gesture")
        << gesture.c_str()
        << osc::EndMessage;

    transmitSocket->Send(packet.Data(), packet.Size());
}