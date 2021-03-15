#ifndef IMAGESTACK_CONTROL_H
#define IMAGESTACK_CONTROL_H
#include "header.h"

class Loop : public Operation {
  public:
    void help();
    void parse(vector<string> args);
};

class Pause : public Operation {
  public:
    void help();
    void parse(vector<string> args);
};

class Time : public Operation {
  public:
    void help();
    void parse(vector<string> args);
};

#include "footer.h"
#endif
