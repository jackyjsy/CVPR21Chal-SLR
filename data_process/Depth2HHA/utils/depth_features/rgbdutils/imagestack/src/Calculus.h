#ifndef IMAGESTACK_CALCULUS_H
#define IMAGESTACK_CALCULUS_H
#include "header.h"

class Gradient : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, string dimensions);
    static void apply(Window im, char dimension);
};

class Integrate : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, string dimensions);
    static void apply(Window im, char dimension);
};

class GradMag : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im);
};

class Poisson : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window dx, Window dy, float termination = 0.01);
};

#include "footer.h"
#endif
