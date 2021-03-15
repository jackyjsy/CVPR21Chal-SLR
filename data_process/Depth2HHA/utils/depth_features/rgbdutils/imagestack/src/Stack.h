#ifndef IMAGESTACK_STACK_H
#define IMAGESTACK_STACK_H
#include "header.h"

// these operations apply only to the stack, so they have no apply method

class Pop : public Operation {
  public:
    void help();
    void parse(vector<string> args);
};

class Push : public Operation {
  public:
    void help();
    void parse(vector<string> args);
};

class Pull : public Operation {
  public:
    void help();
    void parse(vector<string> args);
};

class Dup : public Operation {
  public:
    void help();
    void parse(vector<string> args);
};

#include "footer.h"
#endif
