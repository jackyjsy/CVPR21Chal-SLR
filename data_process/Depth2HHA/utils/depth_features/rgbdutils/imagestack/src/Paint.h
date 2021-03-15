#ifndef IMAGESTACK_PAINT_H
#define IMAGESTACK_PAINT_H
#include "header.h"

class Eval : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, string expression);
};

class EvalChannels : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, vector<string> expressions);
};

class Plot : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int width, int height, float lineThickness);
};

class Composite : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window dst, Window src);
    static void apply(Window dst, Window src, Window mask);
};

#include "footer.h"
#endif
