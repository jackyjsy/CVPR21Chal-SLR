#ifndef IMAGESTACK_HDR_H
#define IMAGESTACK_HDR_H
#include "header.h"

class AssembleHDR : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window frames);
    static Image apply(Window frames, vector<float> &exposures, string gamma="1.0");
  private:
    enum cutoffType { REGULAR, LONGEST_EXPOSURE, SHORTEST_EXPOSURE };
    static float weightFunc(float*, int channels, cutoffType = REGULAR);

};

#include "footer.h"
#endif 
