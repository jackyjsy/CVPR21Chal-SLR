#ifndef IMAGESTACK_COLOR_H
#define IMAGESTACK_COLOR_H
#include "header.h"

class ColorMatrix : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, vector<float> matrix);
};

class ColorConvert : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, string from, string to);
    static Image rgb2hsv(Window im);
    static Image hsv2rgb(Window im);
    static Image rgb2y(Window im);
    static Image y2rgb(Window im);        
    static Image rgb2yuv(Window im);
    static Image yuv2rgb(Window im);
    static Image rgb2xyz(Window im);
    static Image xyz2rgb(Window im);
    static Image lab2xyz(Window im);
    static Image xyz2lab(Window im);
    static Image rgb2lab(Window im);
    static Image lab2rgb(Window im);

    static Image uyvy2yuv(Window im);
    static Image yuyv2yuv(Window im);

    static Image uyvy2rgb(Window im);
    static Image yuyv2rgb(Window im);
};

class Demosaic : public Operation {
public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window win, int xoff, int yoff, bool awb);
};

#include "footer.h"
#endif
