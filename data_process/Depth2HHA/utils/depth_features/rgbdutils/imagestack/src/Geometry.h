#ifndef IMAGESTACK_GEOMETRY_H
#define IMAGESTACK_GEOMETRY_H
#include "header.h"

class Upsample : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int boxWidth, int boxHeight, int boxFrames = 1);
};

class Downsample : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int boxWidth, int boxHeight, int boxFrames = 1);
};

class Subsample : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int boxWidth, int boxHeight, int boxFrames,
                       int offsetX, int offsetY, int offsetT);
    static Image apply(Window im, int boxWidth, int boxHeight, 
                       int offsetX, int offsetY);
    static Image apply(Window im, int boxFrames, int offsetT);
};

class Interleave : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, int rx, int ry, int rt = 1);
};

class Deinterleave : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, int ix, int iy, int it = 1);
};

class Resample : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int width, int height);
    static Image apply(Window im, int width, int height, int frames);
 private:
    static Image resampleT(Window im, int frames);
    static Image resampleX(Window im, int width);
    static Image resampleY(Window im, int height);    
};

class Rotate : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, float degrees);
};

class AffineWarp : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, vector<double> warp);
};

class Crop : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int minX, int minY, int width, int height);
    static Image apply(Window im, int minX, int minY, int minT, int width, int height, int frames);
    static Image apply(Window im);
};

class Flip : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im, char dimension);
};

class Adjoin : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window a, Window b, char dimension);
};

class Transpose : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, char arg1, char arg2);
};

class Translate : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int xoff, int yoff, int toff = 0);
};

class Paste : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window into, Window from,
                      int xdst, int ydst, 
                      int xsrc, int ysrc, 
                      int width, int height);

    static void apply(Window into, Window from,
                      int xdst, int ydst, int tdst = 0);

    static void apply(Window into, Window from, 
                      int xdst, int ydst, int tdst,
                      int xsrc, int ysrc, int tsrc, 
                      int width, int height, int frames);
};

class Tile : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int xTiles, int yTiles, int tTiles = 1);
};

class TileFrames : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int xTiles, int yTiles);
};

class FrameTiles : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int xTiles, int yTiles);
};

class Warp : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window coords, Window source);
};

class Reshape : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im, int newWidth, int newHeight, int newFrames, int newChannels);
    static Image apply(Image im, int newWidth, int newHeight, int newFrames, int newChannels);
};

#include "footer.h"
#endif
