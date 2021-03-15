#include "main.h"
#include "LightField.h"
#include "Geometry.h"
#include "Stack.h"
#include "Convolve.h"
#include "Arithmetic.h"
#include "Color.h"
#include "File.h"
#include "Wavelet.h"
#include "Filter.h" 
#include "LinearAlgebra.h"
#include "header.h"

void LFFocalStack::help() {
    printf("\n-lffocalstack turns a 4d light field into a 3d focal stack. The five arguments\n"
           "are the lenslet width, height, the minimum alpha, the maximum alpha, and the\n"
           "step size between adjacent depths (alpha is slope in line space).\n\n"
           "Usage: ImageStack -load lf.exr -lffocalstack 16 16 -1 1 0.1 -display\n\n");
}

void LFFocalStack::parse(vector<string> args) {
    assert(args.size() == 5, "-lffocalstack takes five arguments.\n");
    LightField lf(stack(0), readInt(args[0]), readInt(args[1]));
    Image im = apply(lf, readFloat(args[2]), readFloat(args[3]), readFloat(args[4]));
    pop();
    push(im);
}


Image LFFocalStack::apply(LightField lf, float minAlpha, float maxAlpha, float deltaAlpha) {
    assert(lf.frames == 1, "Can only turn a single light field into a focal stack\n");

    int frames = 0;
    for (float alpha = minAlpha; alpha <= maxAlpha; alpha += deltaAlpha) frames++;

    Image out(lf.xSize, lf.ySize, frames, lf.channels);

    int t = 0;
    for (float alpha = minAlpha; alpha <= maxAlpha; alpha += deltaAlpha) {
        printf("computing frame %i\n", t+1);
        
        for (int y = 0; y < lf.ySize; y++) {
            for (int v = 0; v < lf.vSize; v++) {
                for (int x = 0; x < lf.xSize; x++) {
                    for (int u = 0; u < lf.uSize; u++) {
                        float fx = x + alpha * (u + 0.5 - lf.uSize * 0.5);
                        float fy = y + alpha * (v + 0.5 - lf.vSize * 0.5);
                        int ix = (int)(fx + 0.5);
                        int iy = (int)(fy + 0.5);
                        if (ix < 0 || iy < 0 || ix >= out.width || iy >= out.height) continue;
                        for (int c = 0; c < out.channels; c++) {
                            out(x, y, t)[c] += lf(ix, iy, u, v)[c];
                        }
                    }
                }
            }
        }
        t++;
    }

    return out;
}


void LFWarp::help() {
    printf("\n-lfwarp treats the top image of the stack as indices (within [0, 1]) into the\n"
           "lightfield represented by the second image, and samples quadrilinearly into it.\n"
           "The two arguments it takes are the width and height of each lenslet.\n"
           "The number of channels in the top image has to be 4, with the channels being\n"
           "the s,t,u and v coordinates in that order.\n"
           "An extra argument of 'quick' at the end switches nearest neighbor resampling on\n"
           "Usage: ImageStack -load lf.jpg -load lfmap.png -lfwarp 8 8 -save out.jpg\n\n");
}

void LFWarp::parse(vector<string> args) {
    assert(args.size() >= 2, "-lfwarp takes at least two arguments.\n");
    assert(stack(0).channels == 4, "Top image for -lfwarp must have 4 channels.\n");
    bool quick=false;
    // parse the rest of the options
    for(unsigned i=2;i<args.size();i++) {
        if(args[i] == "quick")
            quick=true;
    }
    LightField lf(stack(1), readInt(args[0]), readInt(args[1]));
    Image im = apply(lf, stack(0),quick);
    pop();
    pop();
    push(im);
}

Image LFWarp::apply(LightField lf, Window warper, bool quick) {
    // these coordinates are from 0 to 1 each, and 0,0 is lower left corner
    float *coords;
    // these are the LightField object coordinates
    float lx, ly, lu, lv;
    // the output image
    Image out(warper.frames,warper.width,warper.height,lf.channels);
    // do the processing loop
    for(int t=0;t<warper.frames;t++) {
        for(int y=0;y<warper.height;y++) {
            for(int x=0;x<warper.width;x++) {
                coords = warper(x, y, t);
                lx = coords[0]*(lf.xSize-1);
                ly = (1-coords[1])*(lf.ySize-1);
                lu = coords[2]*(lf.uSize-1);
                lv = (1-coords[3])*(lf.vSize-1);
                if(!quick) {
                    lf.sample4D(lx, ly, lu, lv, out(x, y, t));
                } else {
                    int ilx = (int)(lx+0.5);
                    int ily = (int)(ly+0.5);
                    int ilu = (int)(lu+0.5);
                    int ilv = (int)(lv+0.5);
                    ilx = clamp(ilx,0,lf.xSize-1);
                    ily = clamp(ily,0,lf.ySize-1);
                    ilu = clamp(ilu,0,lf.uSize-1);
                    ilv = clamp(ilv,0,lf.vSize-1);
                    float *result = lf(ilx,ily,ilu,ilv);
                    for(int i=0;i<lf.channels;i++)
                        out(x, y, t)[i] = result[i];
                }
            }
        }
    }
    return out;
}


void LFPoint::help() {
    printf("\n-lfpoint colors a single 3d point white in the given light field. The five\n"
           "arguments are the light field u, v, resolution, and then the x, y, and z\n"
           "coordinates of the point. x and y should be in the range [0, 1], while z\n"
           "is disparity. z = 0 will be at the focal plane.\n\n"
           "Usage: ImageStack -load lf.exr -lfpoint 16 16 0.5 0.5 0.1 -save newlf.exr\n\n");
}

void LFPoint::parse(vector<string> args) {
    assert(args.size() == 5, "-lfpoint takes five arguments\n");
    LightField lf(stack(0), readInt(args[0]), readInt(args[1]));
    apply(lf, readFloat(args[2]), readFloat(args[3]), readFloat(args[4]));
}

void LFPoint::apply(LightField lf, float px, float py, float pz) {
    for (int v = 0; v < lf.vSize; v++) {
        for (int u = 0; u < lf.uSize; u++) {
            float pu = u + 0.5 - lf.uSize * 0.5;
            float pv = v + 0.5 - lf.vSize * 0.5;
            // figure out the correct x y
            int x = (int)((px + pz * pu) * lf.xSize + 0.5);
            int y = (int)((py + pz * pv) * lf.ySize + 0.5);
            if (x < 0 || x >= lf.xSize || y < 0 || y >= lf.ySize) continue;
            for (int c = 0; c < lf.channels; c++) {
                lf(x, y, u, v)[c] = 1;
            }
        }
    }
}

#include "footer.h"
