#ifndef IMAGESTACK_IMAGE_H
#define IMAGESTACK_IMAGE_H

#include "tables.h"
#include "header.h"

class Window {
  public:
    Window() {
        xstride = ystride = tstride = width = height = frames = channels = 0;
        data = NULL;
    }

    operator bool() {
        return (data != NULL);
    }
    
    Window(Window im, int minx_, int miny_, int mint_, int width_, int height_, int frames_) {
        int mint = max(0, mint_);
        int maxt = min(im.frames, mint_ + frames_);
        int minx = max(0, minx_);
        int maxx = min(im.width, minx_ + width_);
        int miny = max(0, miny_);
        int maxy = min(im.height, miny_ + height_);

        xstride = im.xstride;
        ystride = im.ystride;
        tstride = im.tstride;

        width = maxx - minx;
        height = maxy - miny;
        frames = maxt - mint;
        channels = im.channels;

        data = im.data + mint * tstride + miny * ystride + minx * xstride;
    }

    bool operator==(const Window &other) const {
        return (data == other.data && 
                width == other.width &&
                height == other.height &&
                frames == other.frames &&
                channels == other.channels);
    }

    float *operator()(int x, int y, int t) {
        return data + x * xstride + y * ystride + t * tstride;
    }

    float *operator()(int x, int y) {
        return data + x * xstride + y * ystride;
    }

    float *operator()(int x) {
        return data + x * xstride;
    }

    typedef enum {ZERO = 0, NEUMANN} BoundaryCondition;

    void sample2D(float fx, float fy, int t, float *result, BoundaryCondition boundary = ZERO) {
        int ix = (int)fx;
        int iy = (int)fy;
        const int LEFT = -2;
        const int RIGHT = 3;
        const int WIDTH = 6;
        int minX = ix + LEFT;
        int maxX = ix + RIGHT;
        int minY = iy + LEFT;
        int maxY = iy + RIGHT;

        float weightX[WIDTH];
        float weightY[WIDTH];
        float totalXWeight = 0, totalYWeight = 0;
        for (int x = 0; x < WIDTH; x++) {
            float diff = (fx - (x + ix + LEFT)); // ranges between +/- RIGHT
            float val = lanczos_3(diff);
            weightX[x] = val;
            totalXWeight += val;
        }

        for (int y = 0; y < WIDTH; y++) {
            float diff = (fy - (y + iy + LEFT)); // ranges between +/- RIGHT
            float val = lanczos_3(diff);
            weightY[y] = val;
            totalYWeight += val;
        }

        totalXWeight = 1.0f/totalXWeight;
        totalYWeight = 1.0f/totalYWeight;

        for (int i = 0; i < WIDTH; i++) {
            weightX[i] *= totalXWeight;
            weightY[i] *= totalYWeight;
        }

        for (int c = 0; c < channels; c++) {
            result[c] = 0;
        }

        if (boundary == NEUMANN) {
        
            float *yWeightPtr = weightY;
            for (int y = minY; y <= maxY; y++) {
                float *xWeightPtr = weightX;
                int sampleY = clamp(0, y, height-1);
                for (int x = minX; x <= maxX; x++) {
                    int sampleX = clamp(0, x, width-1);
                    float yxWeight = (*yWeightPtr) * (*xWeightPtr);
                    float *ptr = (*this)(sampleX, sampleY, t);
                    for (int c = 0; c < channels; c++) {
                        result[c] += ptr[c] * yxWeight;
                    }
                    xWeightPtr++;
                }
                yWeightPtr++;
            }              
        } else {
            float *weightYBase = weightY;
            float *weightXBase = weightX;
            if (minY < 0) {
                weightYBase -= minY;
                minY = 0;
            }
            if (minX < 0) {
                weightXBase -= minX;
                minX = 0;
            }
            if (maxX > width-1) maxX = width-1;
            if (maxY > height-1) maxY = height-1;
            float *yWeightPtr = weightYBase;
            for (int y = minY; y <= maxY; y++) {
                float *xWeightPtr = weightXBase;
                float *imPtr = (*this)(minX, y, t);
                for (int x = minX; x <= maxX; x++) {
                    float yxWeight = (*yWeightPtr) * (*xWeightPtr);
                    for (int c = 0; c < channels; c++) {
                        result[c] += (*imPtr++) * yxWeight;
                    }
                    xWeightPtr++;
                }
                yWeightPtr++;
            }              

        }
    }

    void sample2D(float fx, float fy, float *result) {
        sample2D(fx, fy, 0, result);
    }


    void sample2DLinear(float fx, float fy, float *result) {
        sample2DLinear(fx, fy, 0, result);
    }

    void sample2DLinear(float fx, float fy, int t, float *result) {
        int ix = (int)fx;
        int iy = (int)fy;
        fx -= ix;
        fy -= iy;

        float *ptr = data + t * tstride + iy * ystride + ix * xstride;
        for (int c = 0; c < channels; c++) {
            float s1 = (1-fx) * ptr[c] + fx * ptr[c + xstride];
            float s2 = (1-fx) * ptr[c + ystride] + fx * ptr[c + xstride + ystride];
            result[c] = (1-fy) * s1 + fy * s2;
            
        }

    }

    void sample3DLinear(float fx, float fy, float ft, float *result) {
        int ix = (int)fx;
        int iy = (int)fy;
        int it = (int)ft;
        fx -= ix;
        fy -= iy;
        ft -= it;

        float *ptr = data + it * tstride + iy * ystride + ix * xstride;
        for (int c = 0; c < channels; c++) {
            float s11 = (1-fx) * ptr[c] + fx * ptr[c + xstride];
            float s12 = (1-fx) * ptr[c + ystride] + fx * ptr[c + xstride + ystride];
            float s1 = (1-fy) * s11 + fy * s12;

            float s21 = (1-fx) * ptr[c + tstride] + fx * ptr[c + xstride + tstride];
            float s22 = (1-fx) * ptr[c + ystride + tstride] + fx * ptr[c + xstride + ystride + tstride];
            float s2 = (1-fy) * s21 + fy * s22;

            result[c] = (1-ft) * s1 + ft * s2;           
        }
        
    }

    void sample3D(float fx, float fy, float ft, float *result, BoundaryCondition boundary = ZERO) {
        int ix = (int)fx;
        int iy = (int)fy;
        int it = (int)ft;
        const int LEFT = -2;
        const int RIGHT = 3;
        const int WIDTH = 6;
        int minX = ix + LEFT;
        int maxX = ix + RIGHT;
        int minY = iy + LEFT;
        int maxY = iy + RIGHT;
        int minT = it + LEFT;
        int maxT = it + RIGHT;
        float weightX[WIDTH];
        float weightY[WIDTH];
        float weightT[WIDTH];

        float totalXWeight = 0, totalYWeight = 0, totalTWeight = 0;

        for (int x = 0; x < WIDTH; x++) {
            float diff = (fx - (x + ix + LEFT)); // ranges between +/- RIGHT
            float val = lanczos_3(diff);
            weightX[x] = val;
            totalXWeight += val;
        }

        for (int y = 0; y < WIDTH; y++) {
            float diff = (fy - (y + iy + LEFT)); // ranges between +/- RIGHT
            float val = lanczos_3(diff);
            weightY[y] = val;
            totalYWeight += val;
        }

        for (int t = 0; t < WIDTH; t++) {
            float diff = (ft - (t + it + LEFT)); // ranges between +/- RIGHT
            float val = lanczos_3(diff);
            weightT[t] = val;
            totalTWeight += val;
        }

        totalXWeight = 1.0f/totalXWeight;
        totalYWeight = 1.0f/totalYWeight;
        totalTWeight = 1.0f/totalTWeight;

        for (int i = 0; i < WIDTH; i++) {
            weightX[i] *= totalXWeight;
            weightY[i] *= totalYWeight;
            weightT[i] *= totalTWeight;
        }

        for (int c = 0; c < channels; c++) {
            result[c] = 0;
        }

        if (boundary == NEUMANN) {

            float *tWeightPtr = weightT;
            for (int t = minT; t <= maxT; t++) {
                int sampleT = clamp(t, 0, frames-1);
                float *yWeightPtr = weightY;
                for (int y = minY; y <= maxY; y++) {
                    int sampleY = clamp(y, 0, height-1);
                    float tyWeight = (*yWeightPtr) * (*tWeightPtr);
                    float *xWeightPtr = weightX;
                    for (int x = minX; x <= maxX; x++) {
                        int sampleX = clamp(x, 0, width-1);                                                          
                        float tyxWeight = tyWeight * (*xWeightPtr);
                        float *ptr = (*this)(sampleX, sampleY, sampleT);
                        for (int c = 0; c < channels; c++) {
                            result[c] += ptr[c] * tyxWeight;
                        }
                        xWeightPtr++;
                    }
                    yWeightPtr++;
                }
                tWeightPtr++;
            }    
            
        } else {

            float *weightTBase = weightT;
            float *weightYBase = weightY;
            float *weightXBase = weightX;

            if (minY < 0) {
                weightYBase -= minY;
                minY = 0;
            }
            if (minX < 0) {
                weightXBase -= minX;
                minX = 0;
            }
            if (minT < 0) {
                weightTBase -= minT;
                minT = 0;
            }
            if (maxX > width-1) maxX = width-1;
            if (maxY > height-1) maxY = height-1;
            if (maxT > frames-1) maxT = frames-1;

            float *tWeightPtr = weightTBase;
            for (int t = minT; t <= maxT; t++) {
                float *yWeightPtr = weightYBase;
                for (int y = minY; y <= maxY; y++) {
                    float *xWeightPtr = weightXBase;
                    float *imPtr = (*this)(minX, y, t);
                    for (int x = minX; x <= maxX; x++) {
                        float yxWeight = (*yWeightPtr) * (*xWeightPtr);
                        for (int c = 0; c < channels; c++) {
                            result[c] += (*imPtr++) * yxWeight;
                        }
                        xWeightPtr++;
                    }
                    yWeightPtr++;
                }              
                tWeightPtr++;
            }
        }

    }

    int width, height, frames, channels;
    int xstride, ystride, tstride;
    float *data;    

};

class Image : public Window {
  protected:
    float *memory;
  public:
    Image() : refCount(NULL) {
        width = frames = height = channels = 0;
        xstride = ystride = tstride = 0;
        memory = data = NULL;
    }
    
        
    void debug() {
        printf("%llx(%d@%llx): %d %d %d %d\n", (unsigned long long)data, refCount[0], (unsigned long long)refCount, width, height, frames, channels);
    }
            

    Image(int width_, int height_, int frames_, int channels_, const float *data_ = NULL) {
        width = width_;
        height = height_;
        frames = frames_;
        channels = channels_;

        long long size = ((long long)frames_ * 
                          (long long)height_ *
                          (long long)width_ * 
                          (long long)channels_);
        

        // guarantee 16-byte alignment in case people want to use SSE
        memory = new float[size+3];
        if ((long long)memory & 0xf)
            data = memory + 4 - (((long long)memory & 0xf) >> 2);
        else
            data = memory;
        if (!data_) memset(data, 0, size * sizeof(float));
        else memcpy(data, data_, size * sizeof(float));
        
        xstride = channels;
        ystride = xstride * width;
        tstride = ystride * height;
        refCount = new int;
        *refCount = 1;
        
        //printf("Making new image "); 
        //debug();
    }
    
    // does not copy data

    Image &operator=(const Image &im) {
        if (refCount) {
            refCount[0]--;
            if (*refCount <= 0) {
                delete refCount;
                delete[] memory;
            }
        }

        width = im.width;
        height = im.height;
        channels = im.channels;
        frames = im.frames;
        
        data = im.data;
        memory = im.memory;
        
        xstride = channels;
        ystride = xstride * width;
        tstride = ystride * height;        
        
        refCount = im.refCount;       
        if (refCount) refCount[0]++;
        
        return *this;
    }
    
    Image(const Image &im) {
        width = im.width;
        height = im.height;
        channels = im.channels;
        frames = im.frames;
        
        memory = im.memory;
        data = im.data;       

        xstride = channels;
        ystride = xstride * width;
        tstride = ystride * height;        
        
        refCount = im.refCount;        
        if (refCount) refCount[0]++;       
    }

    // copies data from the window
    Image(Window im) {
        width = im.width;
        height = im.height;
        channels = im.channels;
        frames = im.frames;
        
        xstride = channels;
        ystride = xstride * width;
        tstride = ystride * height;        

        refCount = new int;
        *refCount = 1;
        long long size = ((long long)width *
                          (long long)height *
                          (long long)channels *
                          (long long)frames);

        // guarantee 16-byte alignment in case people want to use SSE
        memory = new float[size+3];
        if ((long long)memory & 0xf)
            data = memory + 4 - (((long long)memory & 0xf) >> 2);
        else
            data = memory;
        
        for (int t = 0; t < frames; t++) {
            for (int y = 0; y < height; y++) {
                float *thisPtr = (*this)(0, y, t);
                float *imPtr = im(0, y, t);
                memcpy(thisPtr, imPtr, sizeof(float)*width*channels);
            }
        }
    }    

    // makes a new copy of this image
    Image copy() {
        return Image(width, height, frames, channels, data);
    }
    
    ~Image();

    int *refCount;
    
  protected:
    Image &operator=(Window im) {
        return *this;
    }
};

#include "footer.h"
#endif
