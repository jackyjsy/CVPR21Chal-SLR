#ifndef IMAGESTACK_PANORAMA_H
#define IMAGESTACK_PANORAMA_H
#include "header.h"

class LoadPanorama : public Operation {
 public:
    void help();

    void parse(vector<string> args);

    static Image apply(string filename, 
                       float minTheta, float maxTheta,
                       float minPhi, float maxPhi,
                       int width, int height);
}; 

class PanoramaBackground : public Operation {
 public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im);
};

#include "footer.h"
#endif
