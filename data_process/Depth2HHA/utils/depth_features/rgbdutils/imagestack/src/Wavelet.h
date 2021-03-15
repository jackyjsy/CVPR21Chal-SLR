#ifndef IMAGESTACK_WAVELET_H
#define IMAGESTACK_WAVELET_H
#include "header.h"

class Haar : public Operation {
 public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, int times = -1);
};

class InverseHaar : public Operation {
 public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, int times = -1);
};

class Daubechies : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im);
};

class InverseDaubechies : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im);
};

#include "footer.h"
#endif
