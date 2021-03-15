#include "main.h"
#include "Network.h"
#include "Exception.h"

#ifdef WIN32
#define close closesocket
#else
#include <sys/errno.h>
#endif

#include "header.h"

void checkInitialized() {    
#ifdef WIN32
    static bool initialized = false;

    if(!initialized) {
        WSADATA WsaDat;
        if(WSAStartup(0x0002, &WsaDat) == 0) {
            initialized = true;
        }
        else {
            panic("Unable to initialize WinSock 2.0\n");
        }
    }
#endif
}

bool isReadable(unsigned int fd, int timeout_ = 0) {
    // perform a select to check if the socket has something for us
    if (timeout_ >= 0) {
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(fd, &fds);
        struct timeval timeout;
        timeout.tv_sec = timeout_ / 1000000;
        timeout.tv_usec = timeout_ % 1000000;
        int ready = select(fd+1, &fds, NULL, NULL, &timeout);
        if (ready == 0) return false;
        if (ready < 0) {
            #ifdef WIN32
            panic("select failed with error %i\n", WSAGetLastError());
            #else 
            panic("select failed with error %i\n", errno);
            #endif
        }
    }
    return true;
}


Address::Address(string hostname_, unsigned short port_) {
    hostname = string(hostname_);
    port = port_;

    checkInitialized();

    struct hostent *host;
    unsigned int ip = 0;
    
    if ((host = gethostbyname(hostname.c_str())) != NULL) {
        memcpy(&ip, host->h_addr, host->h_length);
    } else {
        ip = inet_addr(hostname.c_str());
    }

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = ip;
    addr.sin_port = htons(port);
}

Address::Address(struct sockaddr_in addr_) {
    addr = addr_;
    port = ntohs(addr.sin_port);
    hostname = string(inet_ntoa(addr.sin_addr));
}



TCPConnection::TCPConnection(unsigned short port) {
    checkInitialized();

    // create socket for incoming connections
    int servSock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
    assert(servSock >= 0, "Failed to create socket\n");

    // construct local address
    struct sockaddr_in servAddr;
    memset(&servAddr, 0, sizeof(servAddr));
    servAddr.sin_family = AF_INET;
    servAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servAddr.sin_port = htons(port);

    // bind to the local address
    int result = bind(servSock, (struct sockaddr *)&servAddr, sizeof(servAddr));
    assert(result >= 0, "Failed to bind to port\n");   

    // mark the socket to listen
    result = listen(servSock, 1);
    assert(result >= 0, "Failed to listen\n");

    struct sockaddr_in clntAddr;

    #ifdef WIN32
    int clntLen = sizeof(clntAddr);
    #else
    socklen_t clntLen = sizeof(clntAddr);
    #endif
    int clntSock = accept(servSock, (struct sockaddr *)&clntAddr, &clntLen);
    assert(clntSock >= 0, "Failed to accept\n");

    fd = clntSock;
}

TCPConnection::TCPConnection(Address address) {
    checkInitialized();

    // create a socket
    int sock = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
    assert(sock >= 0, "Failed to create socket\n");
    
    // connect
    int result = connect(sock, (struct sockaddr *)&address.addr, sizeof(address.addr));
    assert(result >= 0, "Failed to connect\n");

    fd = sock;
}

TCPConnection::~TCPConnection() {
    close(fd);
}

bool TCPConnection::recv(char *buffer, int len) {
    checkInitialized();

    int recvBytes = 0;
    while (recvBytes < len) {
        int received = ::recv(fd, buffer + recvBytes, len - recvBytes, 0);
        //assert(received > 0, "recv failed\n");
        if(received <= 0)
          return false;
        recvBytes += received;        
    }
    return true;
}

bool TCPConnection::send(const char *buffer, int len) {
    checkInitialized();

    int sentBytes = ::send(fd, buffer, len, 0);
    /*
    assert(sentBytes >= 0, "send failed\n");
    assert(sentBytes == len, "send sent a different number of bytes than expected\n");
    */
    if(sentBytes != len) return false;
    return true;
}

Image TCPConnection::recvImage() {
    // receive the header
    unsigned int header[4];
    recv((char *)header, 4*sizeof(unsigned int));

    Image im(ntohl(header[0]), ntohl(header[1]), ntohl(header[2]), ntohl(header[3]));

    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            // receive a scanline
            unsigned int *scanline = (unsigned int *)im(0, y, t);
            recv((char *)scanline, im.width * im.channels * sizeof(float));

            // ntohl the scanline
            for (int i = 0; i < im.width * im.channels; i++) {
                scanline[i] = ntohl(scanline[i]);
            }
        }
    }

    return im;
}

void TCPConnection::sendImage(Window im) {
    // Send the header first (4 32-bit unsigned integers)
    unsigned header[4];
    header[0] = htonl(im.width);
    header[1] = htonl(im.height);
    header[2] = htonl(im.frames);
    header[3] = htonl(im.channels);
    
    send((char *)header, sizeof(header));
    
    unsigned int *scanline = new unsigned int[im.width * im.channels];
    
    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            // byte swap a scanline and send it
            memcpy(scanline, im(0, y, t), im.width * im.channels * sizeof(float));

            for (int i = 0; i < im.width * im.channels; i++) {
                scanline[i] = htonl(scanline[i]);
            }
            
            send((char *)scanline, im.width * im.channels * sizeof(float));
        }
    }

    delete[] scanline;
}


namespace UDP {
    int recv(unsigned short port, char *buffer, int maxlen, Address *sender, int timeout) {
        // make a one use server and listen once
        UDPServer serv(port);
        return serv.recv(buffer, maxlen, sender, timeout);
    }

    void send(Address address, const char *buffer, int len) {
        checkInitialized();

        // create a socket
        int sock = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);
        assert(sock >= 0, "Failed to create socket\n");
        
        int sent = sendto(sock, buffer, len, 0, (struct sockaddr *)(&(address.addr)), sizeof(address.addr));
        assert(sent >= 0, "send failed\n");
        assert(sent == len, "send sent a different number of bytes than expected\n"); 
        close(sock);
    }
}


TCPServer::TCPServer(unsigned short port) {
    checkInitialized();

    // create socket for incoming connections
    sock = (int)socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
    assert(sock >= 0, "Failed to create socket\n");

    // construct local address
    struct sockaddr_in servAddr;
    memset(&servAddr, 0, sizeof(servAddr));
    servAddr.sin_family = AF_INET;
    servAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servAddr.sin_port = htons(port);

    // bind to the local address
    int result = bind(sock, (struct sockaddr *)&servAddr, sizeof(servAddr));
    assert(result >= 0, "Failed to bind to port\n");   

    // mark the socket to listen (max 5 incoming connections)
    result = ::listen(sock, 5);
    assert(result >= 0, "Failed to listen\n");
}


TCPServer::~TCPServer() {
    close(sock);
}

TCPConnection *TCPServer::listen(int timeout) {
    checkInitialized();

    if (!isReadable(sock, timeout)) return NULL;

    // accept a connection
    struct sockaddr_in clntAddr;

    #ifdef WIN32
    int clntLen = sizeof(clntAddr);
    #else
    socklen_t clntLen = sizeof(clntAddr);
    #endif

    int clntSock = accept(sock, (struct sockaddr *)&clntAddr, &clntLen);
    assert(clntSock >= 0, "Failed to accept\n");

    TCPConnection *conn = new TCPConnection(); 
    conn->fd = clntSock;
    return conn;
}


UDPServer::UDPServer(unsigned short port) {
    checkInitialized();

    sock = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);
    assert(sock >= 0, "Failed to create socket\n");     
    
    // construct local address
    struct sockaddr_in servAddr;
    memset(&servAddr, 0, sizeof(servAddr));
    servAddr.sin_family = AF_INET;
    servAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servAddr.sin_port = htons(port);
    
    // bind to the local address
    int result = bind(sock, (struct sockaddr *)&servAddr, sizeof(servAddr));
    assert(result >= 0, "Failed to bind to port\n");   
}

UDPServer::~UDPServer() {
    close(sock);
}


int UDPServer::recv(char *buffer, int maxlen, Address *address, int timeout) {

    checkInitialized();

    if (!isReadable(sock, timeout)) return 0;
    struct sockaddr_in sender;
    
    #ifdef WIN32
    int len = sizeof(sender);
    #else
    socklen_t len = sizeof(sender);
    #endif

    int received = ::recvfrom(sock, buffer, maxlen, 0,
                              (struct sockaddr *)&sender, &len);
    //assert(received >= 0, "recv failed\n");
    if(received < 0)
      return received;
    
    // return the address of the sender
    if (address) *address = Address(sender);

    return received;    
}
#include "footer.h"
