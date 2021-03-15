#ifndef IMAGESTACK_NETWORKOPS_H
#define IMAGESTACK_NETWORKOPS_H
#include "header.h"

#include <stdio.h>

class TCPServer;

class Send : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, string host = "127.0.0.1", int port = 5678);
};

class Receive : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(int port = 5678);
    static map<int, TCPServer *> servers;
};

#include "footer.h"
#endif
