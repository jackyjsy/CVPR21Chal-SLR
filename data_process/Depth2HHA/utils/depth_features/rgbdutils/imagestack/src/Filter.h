#ifndef IMAGESTACK_FILTER_H
#define IMAGESTACK_FILTER_H
#include "header.h"

class GaussianBlur : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, float filterWidth, float filterHeight, float filterFrames);
};

class FastBlur : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, float filterWidth, float filterHeight, float filterFrames, bool addMargin = true);
  private:
    // IIR filters
    static void blurX(Window im, float filterWidth, int tapSpacing = 1);
    static void blurY(Window im, float filterHeight, int tapSpacing = 1);
    static void blurT(Window im, float filterFrames, int tapSpacing = 1);

    // helper function for IIR filtering
    static void calculateCoefficients(float sigma, float *c0, float *c1, float *c2, float *c3);
};

class RectFilter : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, int filterWidth, int filterHeight, int filterFrames, int iterations = 1);

  private:
    static void blurX(Window im, int filterSize, int iterations = 1);
    static void blurY(Window im, int filterSize, int iterations = 1);
    static void blurT(Window im, int filterSize, int iterations = 1);
    static void blurXCompletely(Window im);
};


class LanczosBlur : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, float filterWidth, float filterHeight, float filterFrames);
};

class MedianFilter : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int radius);
};

class PercentileFilter : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int radius, float percentile);
};

class CircularFilter : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int radius);
};


class Envelope : public Operation {
public:
    void help();
    void parse(vector<string> args);
    enum Mode {Lower = 0, Upper};
    static Image apply(Window im, Mode m, float smoothness, float edgePreserving);
};

#include "footer.h"
#endif
