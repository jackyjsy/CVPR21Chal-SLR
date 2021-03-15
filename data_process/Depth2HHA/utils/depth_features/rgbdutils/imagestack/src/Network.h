#ifndef IMAGESTACK_NETWORK_H
#define IMAGESTACK_NETWORK_H

#ifdef WIN32
//#include <winsock2.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h> 
#endif

#include <string>
using ::std::string;

#include "macros.h"
#include "Image.h"
#include "header.h"

class Address {
  public:
    Address() {}
    Address(string name_, unsigned short port_);
    Address(struct sockaddr_in addr_);

    struct sockaddr_in addr;
    string hostname;
    unsigned short port;
};

class TCPServer;

class TCPConnection {
  public:
    // connect to a remote port
    TCPConnection(Address address);

    // listen (once) on a local port
    TCPConnection(unsigned short port);

    ~TCPConnection();
    bool recv(char *buffer, int len);
    bool send(const char *buffer, int len);
    
    Image recvImage();
    void sendImage(Window im);

    friend class TCPServer;

  private:
    int fd;
    TCPConnection() {}
};

namespace UDP {
    int recv(unsigned short port, char *buffer, int maxlen, Address *sender = NULL, int timeout = -1); 
    void send(Address address, const char *buffer, int len);
}

class TCPServer {
  public:
    TCPServer(unsigned short port);    
    ~TCPServer();

    // returns a connection or NULL
    TCPConnection *listen(int timeout = -1);    
  private:
    int sock;
};

class UDPServer {
  public:
    UDPServer(unsigned short port);
    ~UDPServer();
    int recv(char *buffer, int maxlen, Address *sender = NULL, int timeout = -1);
  private:
    int sock;
};

#include "footer.h"
#endif
