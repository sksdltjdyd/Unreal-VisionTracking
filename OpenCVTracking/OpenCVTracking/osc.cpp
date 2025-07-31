
#include "osc.h"

OSCSender::OSCSender(const std::string& address, int port) {
    try {
        // UDP 소켓을 통해 OSC 메시지 전송
        transmitSocket = new UdpTransmitSocket(
            IpEndpointName(address.c_str(), port)
        );
        std::cout << "OSC Sender initialized: " << address << ":" << port << std::endl;
    }
    catch (std::exception& e) {
        std::cerr << "OSC 초기화 실패: " << e.what() << std::endl;
        transmitSocket = nullptr;
    }
}

OSCSender::~OSCSender() {
    if (transmitSocket) {
        delete transmitSocket;
    }
}

// 모든 트래킹 데이터를 한 번에 전송
void OSCSender::sendAllTrackingData(
    float mouseX, float mouseY, bool mouseClick,
    float p1X, float p1Y, bool p1Tracking,
    float p2X, float p2Y, bool p2Tracking) {

    if (!transmitSocket) return;

    // OSC 패킷 스트림 생성
    osc::OutboundPacketStream packet(buffer, OUTPUT_BUFFER_SIZE);

    try {
        // OSC 메시지 구성
        // 주소 패턴: /tracking/all
        // 데이터: mouseX, mouseY, mouseClick, p1X, p1Y, p2X, p2Y, p1Track, p2Track
        packet << osc::BeginMessage("/tracking/all")
            << mouseX << mouseY << (mouseClick ? 1 : 0)    // 마우스 데이터
            << p1X << p1Y                                  // Player 1 위치
            << p2X << p2Y                                  // Player 2 위치
            << (p1Tracking ? 1 : 0)                        // Player 1 추적 상태
            << (p2Tracking ? 1 : 0)                        // Player 2 추적 상태
            << osc::EndMessage;

        // 패킷 전송
        transmitSocket->Send(packet.Data(), packet.Size());

    }
    catch (std::exception& e) {
        std::cerr << "OSC 전송 오류: " << e.what() << std::endl;
    }
}


// 개별 데이터 전송 메서드들 (선택적 사용)
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

// 제스처 전송
void OSCSender::sendGesture(const std::string& gesture) {
    if (!transmitSocket) return;

    osc::OutboundPacketStream packet(buffer, OUTPUT_BUFFER_SIZE);
    packet << osc::BeginMessage("/tracking/gesture")
        << gesture.c_str()
        << osc::EndMessage;

    transmitSocket->Send(packet.Data(), packet.Size());
}