#ifndef IMAGESTACK_PROJECTION_H
#define IMAGESTACK_PROJECTION_H
#include "header.h"

class Sinugram : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int directions);
};

#include "footer.h"
#endif
