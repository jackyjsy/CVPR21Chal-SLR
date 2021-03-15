#include "main.h"
#include "Filter.h"
#include "Convolve.h"
#include "Color.h"
#include "Geometry.h"
#include "Arithmetic.h"
#include "header.h"

void GaussianBlur::help() {
    pprintf("-gaussianblur takes a floating point width, height, and frames, and"
            " performs a gaussian blur with those standard deviations. The blur is"
            " performed out to three standard deviations. If given only two"
            " arguments, it performs a blur in x and y only. If given one argument,"
            " it performs the blur in x and y with filter width the same as"
            " height.\n"
            "\n"
            "Usage: ImageStack -load in.jpg -gaussianblur 5 -save blurry.jpg\n\n");
}

void GaussianBlur::parse(vector<string> args) {
    float frames = 0, width = 0, height = 0;
    if (args.size() == 1) {
        width = height = readFloat(args[0]);
    } else if (args.size() == 2) {
        width = readFloat(args[0]);
        height = readFloat(args[1]);
    } else if (args.size() == 3) {
        width  = readFloat(args[0]);
        height = readFloat(args[1]);
        frames = readFloat(args[2]);
    } else {
        panic("-gaussianblur takes one, two, or three arguments\n");
    }

    Image im = apply(stack(0), width, height, frames);
    pop();
    push(im);
}

Image GaussianBlur::apply(Window im, float filterWidth, float filterHeight, float filterFrames) {
    Image out(im);

    if (filterWidth != 0) {
        // make the width filter
        int size = (int)(filterWidth * 6 + 1) | 1;
        // even tiny filters should do something, otherwise we
        // wouldn't have called this function.
        if (size == 1) size = 3;
        int radius = size / 2;
        Image filter(size, 1, 1, 1);
        float sum = 0;
        for (int i = 0; i < size; i++) {
            float diff = (i-radius)/filterWidth;
            float value = expf(-diff * diff / 2);
            filter(i, 0, 0)[0] = value;
            sum += value;
        }
        
        for (int i = 0; i < size; i++) {
            filter(i, 0, 0)[0] /= sum;
        }
        
        out = Convolve::apply(out, filter);
    }

    if (filterHeight != 0) {
        // make the height filter
        int size = (int)(filterHeight * 6 + 1) | 1;
        // even tiny filters should do something, otherwise we
        // wouldn't have called this function.
        if (size == 1) size = 3;
        int radius = size / 2;
        Image filter(1, size, 1, 1);
        float sum = 0;
        for (int i = 0; i < size; i++) {
            float diff = (i-radius)/filterHeight;
            float value = expf(-diff * diff / 2);
            filter(0, i, 0)[0] = value;
            sum += value;
        }
        
        for (int i = 0; i < size; i++) {
            filter(0, i, 0)[0] /= sum;
        }
        
        out = Convolve::apply(out, filter);
    }

    if (filterFrames != 0) {
        // make the frames filter
        int size = (int)(filterFrames * 6 + 1) | 1;       
        // even tiny filters should do something, otherwise we
        // wouldn't have called this function.
        if (size == 1) size = 3;
        int radius = size / 2;
        Image filter(1, 1, size, 1);
        float sum = 0;
        for (int i = 0; i < size; i++) {
            float diff = (i-radius)/filterFrames;
            float value = expf(-diff * diff / 2);
            filter(0, 0, i)[0] = value;
            sum += value;
        }
        
        for (int i = 0; i < size; i++) {
            filter(0, 0, i)[0] /= sum;
        }
        
        out = Convolve::apply(out, filter);
    }

    return out;
}

// This blur implementation was contributed by Tyler Mullen as a
// CS448F project. A competition was held, and this method was found
// to be much faster than other IIRs, filtering by resampling,
// iterated rect filters, and polynomial integral images. The method
// was modified by Andrew Adams to be more ImageStacky (i.e. use
// structures more idiomatic to ImageStack like pointer marching), to
// work for larger sized blurs, and to cover more unusual cases.

void FastBlur::help() {
    pprintf("-fastblur takes a floating point frames, width, and height, and"
            " performs a fast approximate gaussian blur with those standard"
            " deviations using the IIR method of van Vliet et al. If given only two"
            " arguments, it performs a blur in x and y only. If given one argument,"
            " it performs the blur in x and y with filter width the same as"
            " height.\n"
            "\n"
            "Usage: ImageStack -load in.jpg -fastblur 5 -save blurry.jpg\n\n");
}

void FastBlur::parse(vector<string> args) {
    float frames = 0, width = 0, height = 0;
    if (args.size() == 1) {
        width = height = readFloat(args[0]);
    } else if (args.size() == 2) {
        width = readFloat(args[0]);
        height = readFloat(args[1]);
    } else if (args.size() == 3) {
        width  = readFloat(args[0]);
        height = readFloat(args[1]);
        frames = readFloat(args[2]);
    } else {
        panic("-fastblur takes one, two, or three arguments\n");
    }

    apply(stack(0), width, height, frames);
}

void FastBlur::apply(Window im, float filterWidth, float filterHeight, float filterFrames, bool addMargin) {
    assert(filterFrames >= 0 &&
           filterWidth >= 0 &&
           filterHeight >= 0,
           "Filter sizes must be non-negative\n");

    // Prevent filtering in useless directions
    if (im.width == 1)  filterWidth = 0;
    if (im.height == 1) filterHeight = 0;
    if (im.frames == 1) filterFrames = 0;

    // Filter in very narrow directions using the regular Gaussian, as
    // the IIR requires a few pixels to get going. If the Gaussian
    // blur is very narrow, also revert to the naive method, as IIR
    // won't work.
    if (filterFrames > 0 && (im.frames < 16 || filterFrames < 0.5)) {
        Image blurry = GaussianBlur::apply(im, filterFrames, 0, 0);
        FastBlur::apply(blurry, 0, filterWidth, filterHeight);        
        Paste::apply(im, blurry, 0, 0, 0);
        return;
    }

    if (filterWidth > 0 && (im.width < 16 || filterWidth < 0.5)) {
        Image blurry = GaussianBlur::apply(im, 0, filterWidth, 0);
        FastBlur::apply(blurry, filterFrames, 0, filterHeight);        
        Paste::apply(im, blurry, 0, 0, 0);
        return;
    }

    if (filterHeight > 0 && (im.height < 16 || filterHeight < 0.5)) {
        Image blurry = GaussianBlur::apply(im, 0, 0, filterHeight);
        FastBlur::apply(blurry, filterFrames, filterWidth, 0);
        Paste::apply(im, blurry, 0, 0, 0);
        return;
    }

    // IIR filtering fails if the std dev is similar to the image
    // size, because it displays a bias towards the edge values on the
    // starting side. We solve this by adding a margin and using
    // homogeneous weights.
    if (addMargin && (im.frames / filterFrames < 8 ||
                      im.width / filterWidth < 8 ||
                      im.height / filterHeight < 8)) {

        int marginT = (int)(filterFrames);
        int marginX = (int)(filterWidth);
        int marginY = (int)(filterHeight);

        Image bigger(im.width+2*marginX, im.height+2*marginY, im.frames+2*marginT, im.channels+1);
        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {
                float *imPtr = im(0, y, t);
                float *biggerPtr = bigger(marginX, y+marginY, t+marginT);
                for (int x = 0; x < im.width; x++) {
                    *biggerPtr++ = 1;
                    for (int c = 0; c < im.channels; c++) {
                        *biggerPtr++ = *imPtr++;
                    }
                }
            }
        }

        FastBlur::apply(bigger, filterFrames, filterWidth, filterHeight, false);
        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {
                float *imPtr = im(0, y, t);
                float *biggerPtr = bigger(marginX, y+marginY, t+marginT);
                for (int x = 0; x < im.width; x++) {
                    float w = 1.0f/(*biggerPtr++);
                    for (int c = 0; c < im.channels; c++) {
                        *imPtr++ = w*(*biggerPtr++);
                    }                    
                }
            }
        }

        return;        
    }    

    // now perform the blur
    if (filterWidth > 32) {
        // for large filters, we decompose into a dense blur and a
        // sparse blur, by spacing out the taps on the IIR
        float remainingStdDev = sqrtf(filterWidth*filterWidth - 32*32);
        int tapSpacing = (int)(remainingStdDev / 32 + 1);            
        blurX(im, remainingStdDev/tapSpacing, tapSpacing);
        blurX(im, 32, 1);
    } else if (filterWidth > 0) {
        blurX(im, filterWidth, 1);
    }

    if (filterHeight > 32) {
        float remainingStdDev = sqrtf(filterHeight*filterHeight - 32*32);
        int tapSpacing = (int)(remainingStdDev / 32 + 1);            
        blurY(im, remainingStdDev/tapSpacing, tapSpacing);
        blurY(im, 32, 1);
    } else if (filterHeight > 0) {
        blurY(im, filterHeight, 1);
    }

    if (filterFrames > 32) {
        float remainingStdDev = sqrtf(filterFrames*filterFrames - 32*32);
        int tapSpacing = (int)(remainingStdDev / 32 + 1);            
        blurT(im, remainingStdDev/tapSpacing, tapSpacing);
        blurT(im, 32, 1);
    } else if (filterFrames > 0) {
        blurT(im, filterFrames, 1);
    }
}

void FastBlur::blurX(Window im, float sigma, int ts) {
    if (sigma == 0) return;

    float *imPtr;
    // blur in the x-direction
    float c0, c1, c2, c3;
    calculateCoefficients(sigma, &c0, &c1, &c2, &c3);
    
    float invC01 = 1.0f/(c0+c1);
    float invC012 = 1.0f/(c0+c1+c2);
    
    // we step through each row of each frame, and apply a forwards and then
    // a backwards pass of our IIR filter to approximate Gaussian blurring
    // in the x-direction
    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            // forward pass
            
            // use a zero boundary condition in the homogeneous
            // sense (ie zero weight outside the image, divide by
            // the sum of the weights)
            for (int j = 0; j < ts; j++) {
                for (int c = 0; c < im.channels; c++){
                    //im(0*ts+j, y, t)[c] = c0*im(0*ts+j, y, t)[c] / c0;
                    im(ts+j, y, t)[c] = (c0*im(ts+j, y, t)[c] + c1*im(j, y, t)[c]) * invC01;
                    im(2*ts+j, y, t)[c] = (c0*im(2*ts+j, y, t)[c] + c1*im(ts+j, y, t)[c] + c2*im(j, y, t)[c]) * invC012;
                }
            }
            
            
            // now apply the forward filter
            imPtr = im(3*ts, y, t);
            int lookBack = -im.xstride*ts;
            for (int i = 0; i < (im.width-3*ts)*im.channels; i++) {
                // our forwards IIR equation
                imPtr[0] = (c0*imPtr[0] +
                            c1*imPtr[lookBack] +
                            c2*imPtr[lookBack*2] + 
                            c3*imPtr[lookBack*3]);
                imPtr++;
            }
            
            // use a zero boundary condition in the homogeneous
            // sense               
            int x = im.width-3*ts;
            for (int j = 0; j < ts; j++) {
                for (int c = 0; c < im.channels; c++){
                    //im(x+2*ts+j, y, t)[c] = c0*im(x+2*ts+j, y, t)[c] / c0;
                    im(x+ts+j, y, t)[c] = (c0*im(x+ts+j, y, t)[c] + c1*im(x+2*ts+j, y, t)[c]) * invC01;
                    im(x+j, y, t)[c] = (c0*im(x+j, y, t)[c] + c1*im(x+ts+j, y, t)[c] + c2*im(x+2*ts+j, y, t)[c]) * invC012;
                }                                
            }
            
            // backward pass
            imPtr = im(im.width-3*ts, y, t);
            lookBack = im.xstride*ts;
            for (int i = 0; i < (im.width-3*ts)*im.channels; i++) {
                imPtr--;
                imPtr[0] = (c0*imPtr[0] +
                            c1*imPtr[lookBack] +
                            c2*imPtr[lookBack*2] + 
                            c3*imPtr[lookBack*3]);
            }
        }
    } 
}

void FastBlur::blurY(Window im, float sigma, int ts) {
    if (sigma == 0) return;

    float c0, c1, c2, c3;
    calculateCoefficients(sigma, &c0, &c1, &c2, &c3);
    float invC01 = 1.0f/(c0+c1);
    float invC012 = 1.0f/(c0+c1+c2);

    float *imPtr;

    // blur in the y-direction
    //  we do the same thing here as in the x-direction
    //  but we apply im.width different filters in parallel,
    //  for cache coherency's sake, first all going in the "forwards"
    //  direction, and then all going in the "backwards" direction
    for (int t = 0; t < im.frames; t++) {
        // use a zero boundary condition in the homogeneous
        // sense (ie zero weight outside the image, divide by
        // the sum of the weights)
        for (int j = 0; j < ts; j++) {
            for (int x = 0; x < im.width; x++) {
                for (int c = 0; c < im.channels; c++) {                    
                    //im(x, j, t)[c] = c0*im(x, j, t)[c] / c0;
                    im(x, ts+j, t)[c] = (c0*im(x, ts+j, t)[c] + c1*im(x, j, t)[c]) * invC01;
                    im(x, 2*ts+j, t)[c] = (c0*im(x, 2*ts+j, t)[c] + c1*im(x, ts+j, t)[c] + c2*im(x, j, t)[c]) * invC012;
                }
            }
        }
        
        // forward pass
        
        int lookBack = -im.ystride*ts;
        for (int y = 3*ts; y < im.height; y++) {
            imPtr = im(0, y, t);
            for (int i = 0; i < im.width*im.channels; i++) {
                imPtr[0] = (c0*imPtr[0] +
                            c1*imPtr[lookBack] +
                            c2*imPtr[lookBack*2] + 
                            c3*imPtr[lookBack*3]);
                imPtr++;
            }
        }
        
        // use a zero boundary condition in the homogeneous
        // sense (ie zero weight outside the image, divide by
        // the sum of the weights)
        int y = im.height-3*ts;
        for (int j = 0; j < ts; j++) {               
            for (int x = 0; x < im.width; x++) {
                for (int c = 0; c < im.channels; c++) {                    
                    //im(x, y+ts*2+j, t)[c] = c0*im(x, y+ts*2+j, t)[c] / c0;
                    im(x, y+ts+j, t)[c] = (c0*im(x, y+ts+j, t)[c] + c1*im(x, y+ts*2+j, t)[c]) * invC01;
                    im(x, y+j, t)[c] = (c0*im(x, y+j, t)[c] + c1*im(x, y+ts+j, t)[c] + c2*im(x, y+ts*2+j, t)[c]) * invC012;
                }
            }
        }
        
        // backward pass          
        lookBack = im.ystride*ts;
        for (int y = im.height-3*ts-1; y >= 0; y--) {
            imPtr = im(0, y, t);
            for (int i = 0; i < im.width*im.channels; i++) {
                imPtr[0] = (c0*imPtr[0] +
                            c1*imPtr[lookBack] +
                            c2*imPtr[lookBack*2] + 
                            c3*imPtr[lookBack*3]);
                imPtr++;
            }
        }
    }            
}

void FastBlur::blurT(Window im, float sigma, int ts) {
    if (sigma == 0) return;

    float c0, c1, c2, c3;
    calculateCoefficients(sigma, &c0, &c1, &c2, &c3);
    float invC01 = 1.0f/(c0+c1);
    float invC012 = 1.0f/(c0+c1+c2);

    float *imPtr;
    
    // blur in the t-direction 
    // this is the same strategy as blurring in y, but we swap t
    // for y everywhere
    for (int y = 0; y < im.height; y++) {
        // use a zero boundary condition in the homogeneous
        // sense (ie zero weight outside the image, divide by
        // the sum of the weights)
        for (int j = 0; j < ts; j++) {
            for (int x = 0; x < im.width; x++) {
                for (int c = 0; c < im.channels; c++) {                    
                    //im(x, y, j)[c] = c0*im(x, y, j)[c] / c0;
                    im(x, y, ts+j)[c] = (c0*im(x, y, ts+j)[c] + c1*im(x, y, j)[c]) * invC01;
                    im(x, y, 2*ts+j)[c] = (c0*im(x, y, 2*ts+j)[c] + c1*im(x, y, ts+j)[c] + c2*im(x, y, j)[c]) * invC012;
                }
            }
        }
        
        // forward pass
        
        int lookBack = -im.tstride*ts;
        for (int t = 3*ts; t < im.frames; t++) {
            imPtr = im(0, y, t);
            for (int i = 0; i < im.width*im.channels; i++) {
                imPtr[0] = (c0*imPtr[0] +
                            c1*imPtr[lookBack] +
                            c2*imPtr[lookBack*2] + 
                            c3*imPtr[lookBack*3]);
                imPtr++;
            }
        }
        
        // use a zero boundary condition in the homogeneous
        // sense (ie zero weight outside the image, divide by
        // the sum of the weights)
        int t = im.frames-3*ts;
        for (int j = 0; j < ts; j++) {
            for (int x = 0; x < im.width; x++) {
                for (int c = 0; c < im.channels; c++) {                    
                    //im(x, y, t+2*ts+j)[c] = c0*im(x, y, t+2*ts+j)[c] / c0;
                    im(x, y, t+ts+j)[c] = (c0*im(x, y, t+ts+j)[c] + c1*im(x, y, t+2*ts+j)[c]) * invC01;
                    im(x, y, t+j)[c] = (c0*im(x, y, t+j)[c] + c1*im(x, y, t+ts+j)[c] + c2*im(x, y, t+2*ts+j)[c]) * invC012;
                }
            }
        }
        
        // backward pass          
        lookBack = im.tstride*ts;
        for (int t = im.frames-3*ts-1; t >= 0; t--) {
            imPtr = im(0, y, t);
            for (int i = 0; i < im.width*im.channels; i++) {
                imPtr[0] = (c0*imPtr[0] +
                            c1*imPtr[lookBack] +
                            c2*imPtr[lookBack*2] + 
                            c3*imPtr[lookBack*3]);
                imPtr++;
            }
        }            
    }
}


void FastBlur::calculateCoefficients(float sigma, float *c0, float *c1, float *c2, float *c3) {
    // performs the necessary conversion between the sigma of a Gaussian blur
    // and the coefficients used in the IIR filter
    
    float q;

    assert(sigma >= 0.5, "To use IIR filtering, standard deviation of blur must be >= 0.5\n");

    if (sigma < 2.5) {
        q = (3.97156 - 4.14554*sqrtf(1 - 0.26891*sigma));
    } else {
        q = 0.98711*sigma - 0.96330;
    }
    
    float denom = 1.57825 + 2.44413*q + 1.4281*q*q + 0.422205*q*q*q;
    *c1 = (2.44413*q + 2.85619*q*q + 1.26661*q*q*q)/denom;
    *c2 = -(1.4281*q*q + 1.26661*q*q*q)/denom;
    *c3 = (0.422205*q*q*q)/denom;
    *c0 = 1 - (*c1 + *c2 + *c3);   
}



void RectFilter::help() {
    pprintf("-rectfilter performs a iterated rectangular filter on the image. The"
            " four arguments are the filter width, height, frames, and the number of"
            " iterations. If three arguments are given, they are interpreted as"
            " frames, width, and height, and the number of iterations is assumed to"
            " be one. If two arguments are given they are taken as width and height,"
            " and frames is assumed to be one. If one argument is given it is taken"
            " as both width and height, with frames and iterations again assumed to"
            " be one.\n"
            "\n"
           "Usage: ImageStack -load in.jpg -rectfilter 1 10 10 -save out.jpg\n\n");
}

void RectFilter::parse(vector<string> args) {
    int iterations = 1, frames = 1, width = 1, height = 1;
    if (args.size() == 1) {
        width = height = readInt(args[0]);
    } else if (args.size() == 2) {
        width = readInt(args[0]);
        height = readInt(args[1]);
    } else if (args.size() == 3) {
        width = readInt(args[0]);
        height = readInt(args[1]);
        frames = readInt(args[2]);
    } else if (args.size() == 4) {
        width = readInt(args[0]);
        height = readInt(args[1]);
        frames = readInt(args[2]);
        iterations = readInt(args[3]);
    } else {
        panic("-rectfilter takes four or fewer arguments\n");
    }

    apply(stack(0), width, height, frames, iterations);
}

void RectFilter::apply(Window im, int filterWidth, int filterHeight, int filterFrames, int iterations) {
    assert(filterFrames & filterWidth & filterHeight & 1, "filter shape must be odd\n");
    assert(iterations >= 1, "iterations must be at least one\n");

    if (filterFrames != 1) blurT(im, filterFrames, iterations);
    if (filterWidth  != 1) blurX(im, filterWidth, iterations);
    if (filterHeight != 1) blurY(im, filterHeight, iterations);
}

void RectFilter::blurXCompletely(Window im) {    
    double invWidth = 1.0/im.width;
    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {            
            for (int c = 0; c < im.channels; c++) {
                // compute the average for this scanline
                double average = 0;
                float *imPtr = im(0, y, t) + c;
                for (int x = 0; x < im.width; x++) {
                    average += (double)(*imPtr);
                    imPtr += im.xstride;
                }
                average *= invWidth;
                imPtr = im(0, y, t) + c;
                for (int x = 0; x < im.width; x++) {
                    *imPtr = (float)average;
                    imPtr += im.xstride;
                }
            }
        }
    }
}


void RectFilter::blurX(Window im, int width, int iterations) {
    if (width <= 1) return;
    if (im.width == 1) return;

    // special case where the radius is large enough that the image is totally uniformly blurred
    if (im.width <= width/2) {
        blurXCompletely(im);
        return;
    }

    int radius = width/2;
    vector<float> buffer(width);

    vector<float> multiplier(width);
    for (int i = 0; i < width; i++) {
        multiplier[i] = 1.0f/width;
    }

    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            for (int c = 0; c < im.channels; c++) {
                for (int i = 0; i < iterations; i++) {
                    // keep a circular buffer of everything currently inside the kernel
                    // also maintain the sum of this buffer
                    float *ptr = im(0, y, t)+c;
                    
                    double sum = 0;
                    int bufferIndex = 0;
                    int bufferEntries = 0;
                    int stride = im.xstride;

                    // initialize the buffer
                    for (int j = 0; j <= radius; j++) {                        
                        buffer[j] = 0; 
                    }
                    for (int j = radius+1; j < width; j++) {
                        buffer[j] = ptr[(j-radius)*stride];
                        sum += buffer[j];
                        bufferEntries++;
                    }

                    double mult = 1.0/bufferEntries;
                                     
                    // non boundary cases
                    for (int x = 0; x < im.width-radius-1; x++) {
                        // assign the average to the current position
                        ptr[0] = (float)(sum * mult);
                        
                        // move on to the next pixel
                        ptr += stride;
                        
                        // swap out the buffer element, updating the sum
                        float newVal = ptr[radius*stride];
                        sum += newVal - buffer[bufferIndex];
                        buffer[bufferIndex] = newVal;
                        bufferIndex++;
                        if (bufferIndex == width) bufferIndex = 0;

                        if (bufferEntries < width) {
                            bufferEntries++;
                            mult = 1.0/bufferEntries;
                        }
                    }                    
                    
                    // boundary cases
                    for (int x = 0; x < radius+1; x++) {
                        // assign the average to the current position
                        ptr[0] = (float)(sum * mult);
                        
                        // move on to the next pixel
                        ptr += stride;
                        
                        // swap out the buffer element, updating the sum
                        sum -= buffer[bufferIndex];
                        //buffer[bufferIndex] = 0;
                        bufferIndex++;
                        if (bufferIndex == width) bufferIndex = 0;                        

                        bufferEntries--;
                        mult = 1.0/bufferEntries;
                    }
                }
            }
        }
    }    

}

void RectFilter::blurY(Window im, int width, int iterations) {
    if (width <= 1) return;
    if (im.height == 1) return;

    // pull out strips of columns and blur them
    Image chunk(im.height, 8, 1, im.channels);

    for (int t = 0; t < im.frames; t++) {
        for (int x = 0; x < im.width; x += chunk.height) {
            int size = chunk.height;
            if (x + chunk.height >= im.width) size = im.width-x;

            // read into the chunk in a transposed fashion
            for (int y = 0; y < im.height; y++) {
                float *imPtr = im(x, y, t);
                float *chunkPtr = chunk(y, 0);
                for (int j = 0; j < size; j++) { // across
                    for (int c = 0; c < im.channels; c++) {
                        chunkPtr[c] = *imPtr++;
                    }
                    chunkPtr += chunk.ystride;
                }
            }
            
            // blur the chunk
            blurX(chunk, width, iterations);

            // read back from the chunk
            for (int y = 0; y < im.height; y++) {
                float *imPtr = im(x, y, t);
                float *chunkPtr = chunk(y, 0);
                for (int j = 0; j < size; j++) { // across
                    for (int c = 0; c < im.channels; c++) {
                        *imPtr++ = chunkPtr[c];
                    }
                    chunkPtr += chunk.ystride;
                }
            }
        }
    }
}

void RectFilter::blurT(Window im, int width, int iterations) {
    if (width <= 1) return;
    if (im.frames == 1) return;

    // pull out strips across frames from rows and blur them
    Image chunk(im.frames, 8, 1, im.channels);

    for (int y = 0; y < im.height; y++) {
        for (int x = 0; x < im.width; x += chunk.height) {
            int size = chunk.height;
            if (x + chunk.height >= im.width) size = im.width-x;

            // read into the chunk in a transposed fashion
            for (int t = 0; t < im.frames; t++) {
                float *imPtr = im(x, y, t);
                float *chunkPtr = chunk(t, 0);
                for (int j = 0; j < size; j++) { // across
                    for (int c = 0; c < im.channels; c++) {
                        chunkPtr[c] = *imPtr++;
                    }
                    chunkPtr += chunk.ystride;
                }
            }
            
            // blur the chunk
            blurX(chunk, width, iterations);

            // read back from the chunk
            for (int t = 0; t < im.frames; t++) {
                float *imPtr = im(x, y, t);
                float *chunkPtr = chunk(t, 0);
                for (int j = 0; j < size; j++) { // across
                    for (int c = 0; c < im.channels; c++) {
                        *imPtr++ = chunkPtr[c];
                    }
                    chunkPtr += chunk.ystride;
                }
            }
        }
    }
}

void LanczosBlur::help() {
    pprintf("-lanczosblur convolves the current image by a three lobed lanczos"
            " filter. A lanczos filter is a kind of windowed sinc. The three"
            " arguments are filter width, height, and frames. If two arguments are"
            " given, frames is assumed to be one. If one argument is given, it is"
            " interpreted as both width and height.\n"
            "\n"
            "Usage: ImageStack -load big.jpg -lanczosblur 2 -subsample 2 2 0 0 -save small.jpg\n\n");

}

void LanczosBlur::parse(vector<string> args) {
    float frames = 0, width = 0, height = 0;
    if (args.size() == 1) {
        width = height = readFloat(args[0]);
    } else if (args.size() == 2) {
        width = readFloat(args[0]);
        height = readFloat(args[1]);
    } else if (args.size() == 3) {
        width  = readFloat(args[0]);
        height = readFloat(args[1]);
        frames = readFloat(args[2]);
    } else {
        panic("-lanczosblur takes one, two, or three arguments\n");
    }

    Image im = apply(stack(0), width, height, frames);
    pop();
    push(im);
}

Image LanczosBlur::apply(Window im, float filterWidth, float filterHeight, float filterFrames) {
    Image out(im);

    if (filterFrames != 0) {
        // make the frames filter
        int size = (int)(filterFrames * 6 + 1) | 1;
        int radius = size / 2;
        Image filter(1, 1, size, 1);
        float sum = 0;
        for (int i = 0; i < size; i++) {
            float value = lanczos_3((i-radius) / filterFrames);
            filter(0, 0, i)[0] = value;
            sum += value;
        }
        
        for (int i = 0; i < size; i++) {
            filter(0, 0, i)[0] /= sum;
        }
        
        out = Convolve::apply(out, filter);
    }

    if (filterWidth != 0) {
        // make the width filter
        int size = (int)(filterWidth * 6 + 1) | 1;
        int radius = size / 2;
        Image filter(size, 1, 1, 1);
        float sum = 0;
        for (int i = 0; i < size; i++) {
            float value = lanczos_3((i-radius) / filterWidth);
            filter(i, 0, 0)[0] = value;
            sum += value;
        }
        
        for (int i = 0; i < size; i++) {
            filter(i, 0, 0)[0] /= sum;
        }
        
        out = Convolve::apply(out, filter);
    }

    if (filterHeight != 0) {
        // make the height filter
        int size = (int)(filterHeight * 6 + 1) | 1;
        int radius = size / 2;
        Image filter(1, size, 1, 1);
        float sum = 0;
        for (int i = 0; i < size; i++) {
            float value = lanczos_3((i-radius) / filterHeight);
            filter(0, i, 0)[0] = value;
            sum += value;
        }
        
        for (int i = 0; i < size; i++) {
            filter(0, i, 0)[0] /= sum;
        }
        
        out = Convolve::apply(out, filter);
    }

    return out;

}
                        



void MedianFilter::help() {
    printf("-medianfilter applies a median filter with a circular support. The sole argument is\n"
           "the pixel radius of the filter.\n\n"
           "Usage: ImageStack -load input.jpg -median 10 -save output.jpg\n\n");
}

void MedianFilter::parse(vector<string> args) {
    assert(args.size() == 1, "-medianfilter takes one argument\n");
    int radius = readInt(args[0]);
    assert(radius > -1, "radius must be positive");
    Image im = apply(stack(0), radius);
    pop();
    push(im);
}

Image MedianFilter::apply(Window im, int radius) {
    return PercentileFilter::apply(im, radius, 0.5);
}

void PercentileFilter::help() {
    printf("-percentilefilter selects a given statistical percentile over a circular support\n"
           "around each pixel. The two arguments are the support radius, and the percentile.\n"
           "A percentile argument of 0.5 gives a median filter, whereas 0 or 1 give min or\n"
           "max filters.\n\n"
           "Usage: ImageStack -load input.jpg -percentilefilter 10 0.25 -save dark.jpg\n\n");
}

void PercentileFilter::parse(vector<string> args) {
    assert(args.size() == 2, "-percentilefilter takes two arguments\n");
    int radius = readInt(args[0]);
    float percentile = readFloat(args[1]);
    assert(0 <= percentile && percentile <= 1, "percentile must be between zero and one");
    if (percentile == 1) percentile = 0.999;
    assert(radius > -1, "radius must be positive");
    Image im = apply(stack(0), radius, percentile);
    pop();
    push(im);
}

Image PercentileFilter::apply(Window im, int radius, float percentile) {
    // make the histogram
    const int buckets = 256;

    vector<int> histogram(buckets);

    Image out(im.width, im.height, im.frames, im.channels);

    // make the filter edge profile
    vector<int> edge(radius*2+1);

    for (int i = 0; i < 2*radius+1; i++) {
        edge[i] = (int)(sqrtf(radius*radius - (i - radius)*(i-radius)) + 0.0001f);
    }
    
    for (int c = 0; c < im.channels; c++) {
        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {
                // initialize the histogram for this scanline
                for (int i = 0; i < buckets; i++) histogram[i] = 0;
                int lteq = 0;
                int total = 0;
                int medianBucket = buckets/2;

                for (int i = 0; i < 2*radius+1; i++) {
                    int xoff = edge[i];
                    int yoff = i - radius;

                    if (y + yoff >= im.height) break;
                    if (y + yoff < 0) continue;

                    for (int j = 0; j <= xoff; j++) {
                        if (j >= im.width) break;
                        float val = im(j, y+yoff, t)[c];
                        int bucket = HDRtoLDR(val);
                        histogram[bucket]++;
                        if (bucket <= medianBucket) lteq++;
                        total++;
                    }                    
                }
            
                for (int x = 0; x < im.width; x++) {
                    // adjust the median bucket downwards
                    while (lteq > total * percentile && medianBucket > 0) {
                        lteq -= histogram[medianBucket];
                        medianBucket--;
                    }
                    
                    // adjust the median bucket upwards
                    while (lteq <= total * percentile && medianBucket < buckets-1) {
                        medianBucket++;
                        lteq += histogram[medianBucket];
                    }
                    
                    out(x, y, t)[c] = LDRtoHDR(medianBucket);
                    
                    // move the histogram to the right
                    for (int i = 0; i < radius*2+1; i++) {
                        int xoff = edge[i];
                        int yoff = i - radius;
                        
                        if (y + yoff >= im.height) break;
                        if (y + yoff < 0) continue;
                        
                        // subtract old value
                        if (x - xoff >= 0) {
                            float val = im(x-xoff, y+yoff, t)[c];
                            int bucket = HDRtoLDR(val);
                            histogram[bucket]--;
                            if (bucket <= medianBucket) lteq--;
                            total--;
                        }
                        
                        // add new value
                        if (x + xoff + 1 < im.width) {
                            float val = im(x+xoff+1, y+yoff, t)[c];
                            int bucket = HDRtoLDR(val);
                            histogram[bucket]++;
                            if (bucket <= medianBucket) lteq++;
                            total++;
                        }
                    }
                }                
            }
        }
    }

    return out;
}



void CircularFilter::help() {
    printf("\n-circularfilter convolves the image with a uniform circular kernel. It is a good\n"
           "approximate to out of focus blur. The sole argument is the radius of the filter.\n\n"
           "Usage: ImageStack -load in.jpg -circularfilter 10 -save out.jpg\n\n");
}

void CircularFilter::parse(vector<string> args) {
    assert(args.size() == 1, "-circularfilter takes one argument\n");

    Image im = apply(stack(0), readInt(args[0]));
    pop();
    push(im);
}

Image CircularFilter::apply(Window im, int radius) {
    Image out(im.width, im.height, im.frames, im.channels);

    // maintain the average response currently under the filter, and the number of pixels under the filter
    float average = 0;
    int count = 0;

    // make the filter edge profile
    vector<int> edge(radius*2+1);
    for (int i = 0; i < 2*radius+1; i++) {
        edge[i] = (int)(sqrtf(radius*radius - (i - radius)*(i-radius)) + 0.0001f);
    }

    // figure out the filter area
    for (int i = 0; i < 2*radius+1; i++) {
        count += edge[i]*2+1;
    }

    float invArea = 1.0f/count;

    for (int c = 0; c < im.channels; c++) {
        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {
                average = 0;
                // initialize the average and count
                for (int i = 0; i < 2*radius+1; i++) {
                    int xoff = edge[i];
                    int yoff = i - radius;
                    int realY = clamp(y + yoff, 0, im.height-1);

                    for (int x = -xoff; x <= xoff; x++) {
                        int realX = clamp(x, 0, im.width-1);
                        float val = im(realX, realY, t)[c];
                        average += val;
                    }                    
                }
            
                for (int x = 0; x < im.width; x++) {
                    out(x, y, t)[c] = average * invArea;
                    
                    // move the histogram to the right
                    for (int i = 0; i < radius*2+1; i++) {                        
                        int realXOld = max(0, x-edge[i]);
                        int realXNew = min(x+edge[i]+1, im.width-1);
                        int realY = clamp(0, y+i-radius, im.height-1);
                        
                        // add new value, subtract old value
                        average += im(realXNew, realY, t)[c];
                        average -= im(realXOld, realY, t)[c];
                    }
                }                
            }
        }
    }

    return out;
}



void Envelope::help() {
    pprintf("-envelope computes a lower or upper envelope of the input, which is"
            " smooth, and less than (or greater than) the input. The first argument"
            " should be \"lower\" or \"upper\". The second argument is the desired"
            " smoothness, which should be greater than zero and strictly less than"
            " one. The last argument is the degree of edge preserving. If zero, the"
            " output will be smooth everywhere. Larger values produce output that is"
            " permitted to have edges where the input does, in a manner similar to a"
            " bilateral filter.\n"
            "\n"
            "Usage: ImageStack -load a.jpg -envelope upper 0.5 1 -display\n"
            "\n"
            "To locally maximize contrast:\n"
            "ImageStack -load a.jpg -dup -scale 1.1 -envelope lower 0.9 1 -pull 1\n"
            "           -subtract -envelope upper 0.9 1 -offset 1 -pull 1 -pull 2\n"
            "           -add -divide -display\n");
}

void Envelope::parse(vector<string> args) {
    assert(args.size() == 3, "-envelope takes three arguments\n");
    Mode m;
    if (args[0] == "lower") m = Lower;
    else if (args[0] == "upper") m = Upper;
    else panic("Unknown mode: %s. Must be lower or upper.\n", args[0].c_str());

    Image envelope = apply(stack(0), m, readFloat(args[1]), readFloat(args[2]));
    push(envelope);
        
}

Image Envelope::apply(Window im, Mode m, float smoothness, float edgePreserving) {

    Image out(im);

    for (int i = 0; i < 3; i++) {
        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {
                for (int c = 0; c < im.channels; c++) {
                    // forward X pass            
                    float *refPtr = im(0, y, t) + c;
                    float *outPtr = out(0, y, t) + c;
                    float lastIm = *outPtr;
                    float lastOut = lastIm;
                    for (int x = 0; x < im.width; x++) {
                        float thisIm = *outPtr;
                        float alpha = smoothness/(edgePreserving*fabs(thisIm - lastIm) + 1);
                        float thisOut = alpha*lastOut + (1-alpha)*thisIm;
                        if (m == Lower) {
                            if (thisOut > thisIm) thisOut = thisIm;
                        } else {
                            if (thisOut < thisIm) thisOut = thisIm;
                        }
                        *outPtr = thisOut;
                        lastOut = thisOut;
                        lastIm = thisIm;
                        outPtr += out.channels;
                        refPtr += im.channels;
                    }
                    // backward X pass
                    refPtr = im(im.width-1, y, t) + c;
                    outPtr = out(out.width-1, y, t) + c;
                    lastOut = *outPtr;
                    lastIm = lastOut;
                    for (int x = 0; x < im.width; x++) {
                        float thisIm = *outPtr;
                        float alpha = smoothness/(edgePreserving*fabs(thisIm - lastIm) + 1);
                        float thisOut = alpha*lastOut + (1-alpha)*thisIm;
                        if (m == Lower) {
                            if (thisOut > thisIm) thisOut = thisIm;
                        } else {
                            if (thisOut < thisIm) thisOut = thisIm;
                        }
                        *outPtr = thisOut;
                        lastOut = thisOut;
                        lastIm = thisIm;
                        outPtr -= out.channels;
                        refPtr -= im.channels;
                    }                
                }
            }
            
            for (int x = 0; x < im.width; x++) {
                for (int c = 0; c < im.channels; c++) {
                    // forward Y pass
                    float *outPtr = out(x, 0, t) + c;
                    float *refPtr = im(x, 0, t) + c;
                    float lastIm = *outPtr;
                    float lastOut = lastIm;
                    for (int y = 0; y < im.height; y++) {
                        float thisIm = *outPtr;
                        float alpha = smoothness/(edgePreserving*fabs(thisIm - lastIm) + 1);
                        float thisOut = alpha*lastOut + (1-alpha)*thisIm;
                        if (m == Lower) {
                            if (thisOut > thisIm) thisOut = thisIm;
                        } else {
                            if (thisOut < thisIm) thisOut = thisIm;
                        }
                        *outPtr = thisOut;
                        lastOut = thisOut;
                        lastIm = thisIm;
                        outPtr += out.ystride;
                        refPtr += im.ystride;
                    }
                    
                    // backward Y pass
                    outPtr = out(x, out.height-1, t) + c;
                    refPtr = im(x, im.height-1, t) + c;
                    lastIm = *outPtr;
                    lastOut = lastIm;
                    for (int y = 0; y < im.height; y++) {
                        float thisIm = *outPtr;
                        float alpha = smoothness/(edgePreserving*fabs(thisIm - lastIm) + 1);
                        float thisOut = alpha*lastOut + (1-alpha)*thisIm;
                        if (m == Lower) {
                            if (thisOut > thisIm) thisOut = thisIm;
                        } else {
                            if (thisOut < thisIm) thisOut = thisIm;
                        }
                        *outPtr = thisOut;
                        lastOut = thisOut;
                        lastIm = thisIm;
                        outPtr -= out.ystride;
                        refPtr -= im.ystride;
                    }
                }
            }
        }
    }

    return out;
}


#include "footer.h"
