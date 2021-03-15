#ifndef IMAGESTACK_OPERATION_H
#define IMAGESTACK_OPERATION_H
#include "header.h"

class Operation {
  public:
    virtual ~Operation() {};
    virtual void parse(vector<string>) = 0;
    virtual void help() = 0;
};

void loadOperations();
void unloadOperations();

// Meta-operations for getting help and testing

class Help : public Operation {
public:
    void help();
    bool test() {return true;}
    void parse(vector<string> args);
};

#include "footer.h"
#endif
