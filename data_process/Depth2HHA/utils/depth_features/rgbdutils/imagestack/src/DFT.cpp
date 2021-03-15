#ifndef NO_FFTW
#include "main.h"
#include "DFT.h"
#include "Geometry.h"
#include "Stack.h"
#include "Arithmetic.h"
#include "Complex.h"
#include "Display.h"
#include <fftw3.h>
#include "header.h"

void DCT::help() {
    pprintf("-dct performs a real discrete cosine transform on the current"
            " image, over the dimensions given in the argument. If no arguments are"
            " given, every dimension is transformed.\n"
            "\n"
            "Usage: ImageStack -load a.png -dct xy -save freq.png\n");
    
}

void DCT::parse(vector<string> args) {
    assert(args.size() < 2, "-dct takes zero or one argument\n");

    bool x = true, y = true, t = true;
    if (args.size() == 1) {
        x = y = t = false;
        for (size_t i = 0; i < args[0].size(); i++) {
            switch(args[0][i]) {
            case 'x':
                x = true;
                break;
            case 'y':
                y = true;
                break;
            case 't':
                t = true;
                break;
            default:
                panic("Unknown dimension: %c\n", args[0][i]);
                break;
            }
        }

    }
    
    apply(stack(0), x, y, t);
}

void DCT::apply(Window im, bool transformX, bool transformY, bool transformT) {
    if (im.width == 1) transformX = false;
    if (im.height == 1) transformY = false;
    if (im.frames == 1) transformT = false;

    // rank 0
    if (!transformX && !transformY && !transformT) return;

    if (transformX && transformY && transformT) { // rank 3
        int n[] = {im.frames, im.height, im.width};
        int nembed[] = {im.frames, im.tstride/im.ystride, im.ystride/im.xstride};
        fftw_r2r_kind kinds[] = {FFTW_REDFT00, FFTW_REDFT00, FFTW_REDFT00};
        
        fftwf_plan plan = fftwf_plan_many_r2r(3, n, im.channels,
                                              im(0, 0, 0), nembed, im.channels, 1,
                                              im(0, 0, 0), nembed, im.channels, 1,
                                              kinds, FFTW_ESTIMATE);
        
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
    } else if (transformX && transformY) { // rank 2
        int n[] = {im.height, im.width};
        int nembed[] = {im.tstride/im.ystride, im.ystride/im.xstride};
        fftw_r2r_kind kinds[] = {FFTW_REDFT00, FFTW_REDFT00};
            
        for (int t = 0; t < im.frames; t++) {
            fftwf_plan plan = fftwf_plan_many_r2r(2, n, im.channels,
                                                  im(0, 0, t), nembed, im.channels, 1,
                                                  im(0, 0, t), nembed, im.channels, 1,
                                                  kinds, FFTW_ESTIMATE);
            
            fftwf_execute(plan);        
            fftwf_destroy_plan(plan);
        }
    } else if (transformT && transformY) { // rank 2
        int n[] = {im.frames, im.height};
        int nembed[] = {im.frames, im.tstride/im.ystride};
        fftw_r2r_kind kinds[] = {FFTW_REDFT00, FFTW_REDFT00};        

        fftwf_plan plan = fftwf_plan_many_r2r(2, n, im.width*im.channels,
                                              im(0, 0, 0), nembed, im.width*im.channels, 1,
                                              im(0, 0, 0), nembed, im.width*im.channels, 1,
                                              kinds, FFTW_ESTIMATE);
        
        fftwf_execute(plan);        
        fftwf_destroy_plan(plan);
    } else if (transformT && transformX) { // rank 2
        int n[] = {im.frames, im.width};
        int nembed[] = {im.frames, im.tstride/im.xstride};
        fftw_r2r_kind kinds[] = {FFTW_REDFT00, FFTW_REDFT00};        

        for (int y = 0; y < im.height; y++) {        
            fftwf_plan plan = fftwf_plan_many_r2r(2, n, im.channels,
                                                  im(0, y, 0), nembed, im.channels, 1,
                                                  im(0, y, 0), nembed, im.channels, 1,
                                                  kinds, FFTW_ESTIMATE);
            
            fftwf_execute(plan);        
            fftwf_destroy_plan(plan);        
        }
    } else if (transformX) { // rank 1
        int n[] = {im.width};
        int nembed[] = {im.width};
        fftw_r2r_kind kinds[] = {FFTW_REDFT00};        

        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {        
                fftwf_plan plan = fftwf_plan_many_r2r(1, n, im.channels,
                                                      im(0, y, t), nembed, im.channels, 1,
                                                      im(0, y, t), nembed, im.channels, 1,
                                                      kinds, FFTW_ESTIMATE);
                
                fftwf_execute(plan);        
                fftwf_destroy_plan(plan);        
            }        
        }
    } else if (transformY) { // rank 1
        int n[] = {im.height};
        int nembed[] = {im.height};
        fftw_r2r_kind kinds[] = {FFTW_REDFT00};        

        for (int t = 0; t < im.frames; t++) {
            fftwf_plan plan = fftwf_plan_many_r2r(1, n, im.width*im.channels,
                                                  im(0, 0, t), nembed, im.ystride, 1,
                                                  im(0, 0, t), nembed, im.ystride, 1,
                                                  kinds, FFTW_ESTIMATE);
            
            fftwf_execute(plan);        
            fftwf_destroy_plan(plan);        
        }        
    } else if (transformT) { // rank 1
        int n[] = {im.frames};
        int nembed[] = {im.frames};
        fftw_r2r_kind kinds[] = {FFTW_REDFT00};        

        for (int y = 0; y < im.height; y++) {
            fftwf_plan plan = fftwf_plan_many_r2r(1, n, im.width*im.channels,
                                                  im(0, y, 0), nembed, im.tstride, 1,
                                                  im(0, y, 0), nembed, im.tstride, 1,
                                                  kinds, FFTW_ESTIMATE);
            fftwf_execute(plan);        
            fftwf_destroy_plan(plan);        
        }
    }       

    float m = 1.0;
    if (transformX) m *= im.width;
    if (transformY) m *= im.height;
    if (transformT) m *= im.frames;
    Scale::apply(im, 1.0f/sqrtf(m));
    

}

void FFT::help() {
    pprintf("-fft performs a fast dft on the current image, whose values"
            " are interpreted as complex. The input is an image with 2*c channels,"
            " where channel 2*i is the real part of the i\'th channel, and channel"
            " 2*i+1 is the imaginary part of the i'th channel. The output image is"
            " laid out the same way.\n"
            "\n"
            "Usage: ImageStack -load a.tmp -fftcomplex -save freq.tmp\n\n");
    
}

void FFT::parse(vector<string> args) {
    assert(args.size() < 2, "-fft takes zero or one argument\n");

    bool x = true, y = true, t = true;
    if (args.size() == 1) {
        x = y = t = false;
        for (size_t i = 0; i < args[0].size(); i++) {
            switch(args[0][i]) {
            case 'x':
                x = true;
                break;
            case 'y':
                y = true;
                break;
            case 't':
                t = true;
                break;
            default:
                panic("Unknown dimension: %c\n", args[0][i]);
                break;
            }
        }

    }
    
    apply(stack(0), x, y, t);
}

void FFT::apply(Window im, bool transformX, bool transformY, bool transformT, bool inverse) {
    assert(im.channels % 2 == 0, "-fft requires an image with an even number of channels\n");

    if (im.width == 1) transformX = false;
    if (im.height == 1) transformY = false;
    if (im.frames == 1) transformT = false;

    // rank 0
    if (!transformX && !transformY && !transformT) return;

    if (transformX && transformY && transformT) { // rank 3
        int n[] = {im.frames, im.height, im.width};
        int nembed[] = {im.frames, im.tstride/im.ystride, im.ystride/im.xstride};
        
        fftwf_plan plan = fftwf_plan_many_dft(3, n, im.channels/2,
                                              (fftwf_complex *)im(0, 0, 0), nembed, im.channels/2, 1,
                                              (fftwf_complex *)im(0, 0, 0), nembed, im.channels/2, 1,
                                              inverse ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_ESTIMATE);
        
        fftwf_execute(plan);
        fftwf_destroy_plan(plan);
    } else if (transformX && transformY) { // rank 2
        int n[] = {im.height, im.width};
        int nembed[] = {im.tstride/im.ystride, im.ystride/im.xstride};
            
        for (int t = 0; t < im.frames; t++) {
            fftwf_plan plan = fftwf_plan_many_dft(2, n, im.channels/2,
                                                  (fftwf_complex *)im(0, 0, t), nembed, im.channels/2, 1,
                                                  (fftwf_complex *)im(0, 0, t), nembed, im.channels/2, 1,
                                                  inverse ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_ESTIMATE);
            
            fftwf_execute(plan);        
            fftwf_destroy_plan(plan);
        }
    } else if (transformT && transformY) { // rank 2
        int n[] = {im.frames, im.height};
        int nembed[] = {im.frames, im.tstride/im.ystride};
        
        fftwf_plan plan = fftwf_plan_many_dft(2, n, im.width*im.channels/2,
                                              (fftwf_complex *)im(0, 0, 0), nembed, im.width*im.channels/2, 1,
                                              (fftwf_complex *)im(0, 0, 0), nembed, im.width*im.channels/2, 1,
                                              inverse ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_ESTIMATE);
        
        fftwf_execute(plan);        
        fftwf_destroy_plan(plan);
    } else if (transformT && transformX) { // rank 2
        int n[] = {im.frames, im.width};
        int nembed[] = {im.frames, im.tstride/im.xstride};

        for (int y = 0; y < im.height; y++) {        
            fftwf_plan plan = fftwf_plan_many_dft(2, n, im.channels/2,
                                                  (fftwf_complex *)im(0, y, 0), nembed, im.channels/2, 1,
                                                  (fftwf_complex *)im(0, y, 0), nembed, im.channels/2, 1,
                                                  inverse ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_ESTIMATE);
            
            fftwf_execute(plan);        
            fftwf_destroy_plan(plan);        
        }
    } else if (transformX) { // rank 1
        int n[] = {im.width};
        int nembed[] = {im.width};

        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {        
                fftwf_plan plan = fftwf_plan_many_dft(1, n, im.channels/2,
                                                      (fftwf_complex *)im(0, y, t), nembed, im.channels/2, 1,
                                                      (fftwf_complex *)im(0, y, t), nembed, im.channels/2, 1,
                                                      inverse ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_ESTIMATE);
                
                fftwf_execute(plan);        
                fftwf_destroy_plan(plan);        
            }        
        }
    } else if (transformY) { // rank 1
        int n[] = {im.height};
        int nembed[] = {im.height};

        for (int t = 0; t < im.frames; t++) {
            fftwf_plan plan = fftwf_plan_many_dft(1, n, im.width*im.channels/2,
                                                  (fftwf_complex *)im(0, 0, t), nembed, im.ystride/2, 1,
                                                  (fftwf_complex *)im(0, 0, t), nembed, im.ystride/2, 1,
                                                  inverse ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_ESTIMATE);
            
            fftwf_execute(plan);        
            fftwf_destroy_plan(plan);        
        }        
    } else if (transformT) { // rank 1
        int n[] = {im.frames};
        int nembed[] = {im.frames};
        for (int y = 0; y < im.height; y++) {
            fftwf_plan plan = fftwf_plan_many_dft(1, n, im.width*im.channels/2,
                                                  (fftwf_complex *)im(0, y, 0), nembed, im.tstride/2, 1,
                                                  (fftwf_complex *)im(0, y, 0), nembed, im.tstride/2, 1,
                                                  inverse ? FFTW_BACKWARD : FFTW_FORWARD, FFTW_ESTIMATE);
            fftwf_execute(plan);        
            fftwf_destroy_plan(plan);        
        }
    }       

    if (inverse) {
        float m = 1.0;
        if (transformX) m *= im.width;
        if (transformY) m *= im.height;
        if (transformT) m *= im.frames;
        Scale::apply(im, 1.0f/m);
    }
    

}


void IFFT::help() {
    pprintf("-ifft performs an inverse dft on the current image, whose values are"
            " complex. The input and output are images with 2*c channels, where"
            " channel 2*i is the real part of the i\'th channel, and channel 2*i+1"
            " is the imaginary part of the i'th channel.\n"
            "\n"
            "Usage: ImageStack -load a.tga -fftcomplex -save freq.tga\n\n");
}

void IFFT::parse(vector<string> args) {
    assert(args.size() < 2, "-ifft takes zero or one argument\n");

    bool x = true, y = true, t = true;
    if (args.size() == 1) {
        x = y = t = false;
        for (size_t i = 0; i < args[0].size(); i++) {
            switch(args[0][i]) {
            case 'x':
                x = true;
                break;
            case 'y':
                y = true;
                break;
            case 't':
                t = true;
                break;
            default:
                panic("Unknown dimension: %c\n", args[0][i]);
                break;
            }
        }
    }
    
    apply(stack(0), x, y, t);
}


void IFFT::apply(Window im, bool x, bool y, bool t) {
    FFT::apply(im, x, y, t, true);
}




void FFTConvolve::help() {
    pprintf("-fftconvolve performs convolution in Fourier space. It is much faster"
            " than -convolve for large kernels. The two arguments are the boundary"
            " condition (zero, clamp, wrap, homogeneous) and the vector-vector"
            " multiplication used (inner, outer, elementwise). The defaults are wrap"
            " and outer respectively. See -convolve for a description of each"
            " option.\n"
            "\n"
            "Usage: ImageStack -load filter.tmp -load im.jpg -fftconvolve zero inner\n");
}

void FFTConvolve::parse(vector<string> args) {
    Multiply::Mode m;
    Convolve::BoundaryCondition b;

    if (args.size() > 0) {
        if (args[0] == "zero") b = Convolve::Zero;
        else if (args[0] == "homogeneous") b = Convolve::Homogeneous;
        else if (args[0] == "clamp") b = Convolve::Clamp;
        else if (args[0] == "wrap") b = Convolve::Wrap;
        else {
            panic("Unknown boundary condition: %s\n", args[0].c_str());
        }
    } else {
        b = Convolve::Wrap;
    }

    if (args.size() > 1) {
        if (args[1] == "inner") m = Multiply::Inner;
        else if (args[1] == "outer") m = Multiply::Outer;
        else if (args[1] == "elementwise") m = Multiply::Elementwise;
        else {
            panic("Unknown vector-vector multiplication: %s\n", args[1].c_str());
        }
    } else {
        m = Multiply::Outer;
    }

    Image im = apply(stack(0), stack(1), b, m);
    pop();
    push(im);
}

Image FFTConvolve::apply(Window im, Window filter, Convolve::BoundaryCondition b, Multiply::Mode m) {

    int resultChannels = 0;

    // check the number of channels is correct
    if (m == Multiply::Inner) {
        assert(im.channels % filter.channels == 0 || filter.channels % im.channels == 0,
               "For inner-product convolution either the image must have a number of"
               " channels that is a multiple of the number of channels in the filter,"
               " or vice-versa.\n");
        resultChannels = max(im.channels / filter.channels, filter.channels / im.channels);
    } else if (m == Multiply::Outer) {
        // anything goes
        resultChannels = im.channels * filter.channels;
    } else if (m == Multiply::Elementwise) {
        assert(im.channels == filter.channels, 
               "For elementwise convolution the filter must have the same number of channels as the image\n");
        resultChannels = im.channels;
    } else {
        panic("Unknown channel mode: %d\n", m);
    }

    // Deal with the homogeneous case recursively. This is slightly
    // inefficient because we construct and transform the filter
    // twice, but it makes the code much simpler
    if (b == Convolve::Homogeneous) {
        Image result = apply(im, filter, Convolve::Zero, m);
        Image weight(im.width, im.height, im.frames, im.channels);
        Offset::apply(weight, 1.0f);
        Image resultW = apply(weight, filter, Convolve::Zero, m);
        Divide::apply(result, resultW);
        return result;
    }
    
    assert(filter.width % 2 == 1 &&
           filter.height % 2 == 1 &&
           filter.frames % 2 == 1,
           "The filter must have odd dimensions\n");

    int xPad = filter.width/2;
    int yPad = filter.height/2;
    int tPad = filter.frames/2;

    if (b == Convolve::Wrap) {
        xPad = yPad = tPad = 0;
    }

    Image imT;
    Image weightT;

    imT = Image(im.width+xPad*2, im.height+yPad*2, im.frames+tPad*2, im.channels*2);
    
    //printf("1\n"); fflush(stdout);
    // 1) Make the padded complex image
    if (b == Convolve::Clamp) {
        for (int t = 0; t < imT.frames; t++) {
            int st = clamp(t-tPad, 0, im.frames-1);
            for (int y = 0; y < imT.height; y++) {
                int sy = clamp(y-yPad, 0, im.height-1);
                float *imTPtr = imT(0, y, t);
                float *imPtr = im(0, sy, st);
                for (int x = 0; x < imT.width; x++) {
                    int sx = clamp(x-xPad, 0, im.width-1);
                    for (int c = 0; c < im.channels; c++) {
                        *imTPtr++ = imPtr[sx*im.channels+c];
                        *imTPtr++ = 0;
                    }
                }
            }
        }
    } else { // Zero or Wrap
        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {
                float *imPtr = im(0, y, t);
                float *imTPtr = imT(xPad, y+yPad, t+tPad);
                for (int x = 0; x < im.width; x++) {                    
                    for (int c = 0; c < im.channels; c++) {
                        *imTPtr++ = *imPtr++;
                        imTPtr++;
                    }
                }
            }
        }
    }
    
    //printf("2\n"); fflush(stdout);
    // 2) Transform the padded image
    FFT::apply(imT);

    //printf("3\n"); fflush(stdout);
    // 3) Make a padded complex filter of the same size
    Image filterT(imT.width, imT.height, imT.frames, filter.channels*2);
    for (int t = 0; t < filter.frames; t++) {
        int ft = t - filter.frames/2;
        if (ft < 0) ft += filterT.frames;
        for (int y = 0; y < filter.height; y++) {
            int fy = y - filter.height/2;
            if (fy < 0) fy += filterT.height;
            for (int x = 0; x < filter.width; x++) {
                for (int c = 0; c < filter.channels; c++) {
                    int fx = x - filter.width/2;
                    if (fx < 0) fx += filterT.width;
                    filterT(fx, fy, ft)[2*c] = filter(x, y, t)[c];
                }
            }
        }
    }
    
    //printf("4\n"); fflush(stdout);
    // 4) Transform the padded filter
    FFT::apply(filterT);

    //printf("5\n"); fflush(stdout);
    // 5) Multiply the two into a padded complex transformed result
    Image resultT(imT.width, imT.height, imT.frames, resultChannels*2);

    for (int t = 0; t < resultT.frames; t++) {
        for (int y = 0; y < resultT.height; y++) {
            float *resultTPtr = resultT(0, y, t);
            float *filterTPtr = filterT(0, y, t);
            float *imTPtr     = imT(0, y, t);
            if (m == Multiply::Outer) {
                for (int x = 0; x < resultT.width; x++) {
                    for (int cf = 0; cf < filterT.channels; cf+=2) {                            
                        for (int ci = 0; ci < imT.channels; ci+=2) {
                            *resultTPtr++ = filterTPtr[cf]*imTPtr[ci] - filterTPtr[cf+1]*imTPtr[ci+1];
                            *resultTPtr++ = filterTPtr[cf+1]*imTPtr[ci] + filterTPtr[cf]*imTPtr[ci+1];
                        }
                    }
                    imTPtr     += imT.channels;
                    filterTPtr += filterT.channels;
                }
            } else if (m == Multiply::Inner && filter.channels > im.channels) {
                for (int x = 0; x < resultT.width; x++) {
                    for (int cr = 0; cr < resultChannels; cr++) {
                        for (int ci = 0; ci < imT.channels; ci+=2) {
                            resultTPtr[0] += filterTPtr[0]*imTPtr[ci] - filterTPtr[1]*imTPtr[ci+1];
                            resultTPtr[1] += filterTPtr[1]*imTPtr[ci] + filterTPtr[0]*imTPtr[ci+1];
                            filterTPtr += 2;
                        }
                        resultTPtr += 2;
                    }
                    imTPtr += imT.channels;
                }
            } else if (m == Multiply::Inner) {
                for (int x = 0; x < resultT.width; x++) {
                    for (int cr = 0; cr < resultChannels; cr++) {
                        for (int cf = 0; cf < filterT.channels; cf+=2) {
                            resultTPtr[0] += filterTPtr[cf]*imTPtr[0] - filterTPtr[cf+1]*imTPtr[1];
                            resultTPtr[1] += filterTPtr[cf+1]*imTPtr[0] + filterTPtr[cf]*imTPtr[1];
                            imTPtr += 2;
                        }
                        resultTPtr += 2;
                    }
                    filterTPtr += filterT.channels;
                }
            } else { // m == ELEMENTWISE
                for (int x = 0; x < resultT.width; x++) {
                    for (int c = 0; c < resultChannels; c++) {
                        resultTPtr[0] += filterTPtr[0]*imTPtr[0] - filterTPtr[1]*imTPtr[1];
                        resultTPtr[1] += filterTPtr[1]*imTPtr[0] + filterTPtr[0]*imTPtr[1];
                        imTPtr += 2;                        
                        resultTPtr += 2;
                        filterTPtr += 2;
                    }
                }
            }
        }
    }

    //printf("6\n"); fflush(stdout);
    // 6) Inverse transorm the result
    IFFT::apply(resultT);
 
    //printf("7\n"); fflush(stdout);
    // 7) Remove the padding, and convert back to real numbers
    Image result(im.width, im.height, im.frames, resultChannels);
    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            float *resultPtr = result(0, y, t);
            float *resultTPtr = resultT(xPad, y+yPad, t+tPad);
            for (int x = 0; x < im.width; x++) {
                for (int c = 0; c < resultChannels; c++) {
                    *resultPtr++ = *resultTPtr++;
                    // skip the imaginary part
                    resultTPtr++;
                }
            }
        }
    }
    //printf("8\n"); fflush(stdout);
    return result;
}


void FFTDeconvolve::help() {
    pprintf("\n-fftdeconvolve uses Fourier-space math to undo a convolution using"
            " the gaussian image prior described in Levin et al. 2007. The"
            " convolution specified must be 2D. The arguments are the weight"
            " argument (the weight assigned to the gradients being minimized), then"
            " the filter width, filter height, and filter frames, then the filter"
            " to be deconvolved by in row major form. With only the first argument,"
            " fftdeconvolve will use the next image on the stack as the filter. It"
            " must be single channel.\n"
            "\n"
            "Usage: ImageStack -load filter.png -load in.jpg -fftdeconvolve 0.01 -save dcv.jpg\n\n");
}

void FFTDeconvolve::parse(vector<string> args) {
    if (args.size() > 4) {
        int width, height, frames, size;
        float weight = readFloat(args[0]);
        width = readInt(args[2]);
        height = readInt(args[3]);
        frames = readInt(args[1]);
        size = frames * width * height;
        assert((int)args.size() == size + 4, "a size of %ix%ix%i requires %i more arguments. %i were given.", 
               frames, width, height, size, (int)args.size() - 4);
        assert(size % 2 == 1, "matrix must have odd size\n");
        
        Image filter(frames, width, height, 1);

        size_t i = 4;
        for (int t = 0; t < frames; t++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    filter(t, x, y)[0] = readFloat(args[i++]);
                }
            }
        }        
        
        Image im = apply(stack(0), filter, weight);
        pop();
        push(im);
    } else if (args.size() == 1) {
        assert(stack(1).channels == 1, "filter must be single channel\n");
        float seconds = readFloat(args[0]);
        Image im = apply(stack(0), stack(1), seconds);
        pop();
        push(im);
    } else {
        panic("-fftdeconvolve needs either one or at least five arguments\n");
    }
}

// Frequency-space gaussian deconvolution is pretty simple (outside of the fourier transforms) 
// Output = fft(filter) * fft(image) / (fft(filter)^2 + weight*fft(gradient filters)^2)
Image FFTDeconvolve::apply(Window im, Window filter, float weight) {

    assert(im.frames == 1, "-fftdeconvolve can only handle single-frame images\n");
    assert(filter.frames == 1, "-fftdeconvolve can only handle single-frame filters\n");
    assert(filter.channels == 1, "-fftdeconvolve can only handle single-channel filters\n");

    // sum of second derivatives filter
    Image fft_g(im.width, im.height, 1, 2);
    fft_g(0, 0)[0] = weight;
    fft_g(im.width-1, 0)[0] = -weight*0.25;
    fft_g(0, im.height-1,0)[0] = -weight*0.25;
    fft_g(1, 0)[0] = -weight*0.25;
    fft_g(0, 1)[0] = -weight*0.25;
    FFT::apply(fft_g);

    Image fft_im = RealComplex::apply(im);
    FFT::apply(fft_im);

    Image fft_filter(im.width, im.height, 1, 2);
    for (int y = 0; y < filter.height; y++) {
        int fy = y - filter.height/2;
        if (fy < 0) fy += fft_filter.height;
        for (int x = 0; x < filter.width; x++) {
            for (int c = 0; c < filter.channels; c++) {
                int fx = x - filter.width/2;
                if (fx < 0) fx += fft_filter.width;
                fft_filter(fx, fy)[2*c] = filter(x, y)[c];
            }
        }
    }
    FFT::apply(fft_filter);

    ComplexMultiply::apply(fft_im, fft_filter, true); 
    ComplexMultiply::apply(fft_filter, fft_filter, true); 

    Add::apply(fft_filter, fft_g); 
    ComplexDivide::apply(fft_im, fft_filter, false); 

    IFFT::apply(fft_im);
    return ComplexReal::apply(fft_im);
}

void FFTPoisson::help() {
    printf("-fftpoisson computes an image from a gradient field in the same way as"
           " -poisson. It interprets the top image on the stack as the y gradient,"
           " and the next image as the x gradient. If a single argument is given,"
           " it uses that as a weight, and interprets the third image on the stack"
           " as a rough target output. The output of this operation will adhere to"
           " the target proportionally to the given weight.\n"
           "\n"
           "Usage: ImageStack -load gx.tmp -load gy.tmp -fftpoisson -display\n");
}


void FFTPoisson::parse(vector<string> args) {
    Image im;

    if (args.size() == 0) {
        im = apply(stack(1), stack(0), Window(), 0);
    } else if (args.size() == 1) {
        im = apply(stack(1), stack(0), stack(2), readFloat(args[0]));
    } else {
        panic("-fftpoisson takes zero or one arguments\n");
    }

    push(im);
}

// This implementation was based on code by Pravin Bhat and is
// available at:
// http://grail.cs.washington.edu/projects/screenedPoissonEq/ Bhat P.,
// Curless B., Cohen M., and Zitnick L. Fourier Analysis of the 2D
// Screened Poisson Equation for Gradient Domain Problems. European
// Conference on Computer Vision (ECCV) 2008.

// It was modified for ImageStack by Neeraj Agrawal and Ritvik Mudur,
// and further modified by Andrew Adams to change the boundary
// conditions expected on the gradient images (ImageStack uses zero
// boundary conditions on gradient images).

Image FFTPoisson::apply(Window dx, Window dy, Window target, float targetStrength) {
    
    assert(dx.width == dy.width &&
           dx.height == dy.height &&
           dx.frames == dy.frames &&
           dx.channels == dy.channels,
           "x gradient must be same size as y gradient\n");
    if (target) {
        assert(target.width == dx.width &&
               target.height == dx.height &&
               target.frames == dx.frames &&
               target.channels == dx.channels,
               "target image must have the same size as the gradient images\n");               
    }

    Image fftBuff(dx.width, dx.height, dx.frames, 1);

    //compute two 1D lookup tables for computing the DCT of a 2D Laplacian on the fly    
    Image ftLapY(1, dx.height, 1, 1);
    Image ftLapX(dx.width, 1, 1, 1);

    for(int x = 0; x < dx.width; x++)
        ftLapX(x, 0)[0] = 2.0f * cos((M_PI * x) / (dx.width - 1));
    
    for(int y = 0; y < dx.height; y++)
        ftLapY(0, y)[0] = -4.0f + (2.0f * cos((M_PI * y) / (dx.height - 1)));
    
    // Create a DCT-I plan, which is its own inverse.
    fftwf_plan fftPlan;        
    fftPlan = fftwf_plan_r2r_2d(dx.height, dx.width, 
                                fftBuff(0, 0), fftBuff(0, 0), 
                                FFTW_REDFT00, FFTW_REDFT00, FFTW_ESTIMATE); //use FFTW_PATIENT when plan can be reused
    
    Image out(dx.width, dx.height, dx.frames, dx.channels);

    for (int t = 0; t < dx.frames; t++) {
        for (int c = 0; c < dx.channels; c++) {            

            float dcSum = 0.0f;
            
            // compute h_hat from u, gx, gy (see equation 48 in the paper), as well as the DC term of u's DCT.
            float *fftPtr = fftBuff(0, 0);
            for(int y = 0; y < dx.height; y++) {
                for(int x = 0; x < dx.width; x++) {
                    // Compute DC term of u's DCT without computing the whole DCT.
                    float dcMult = 1.0f;
                    if ((x > 0) && (x < dx.width  - 1))
                        dcMult *= 2.0f;
                    if ((y > 0) && (y < dx.height - 1))
                        dcMult *= 2.0f;
                    
                    if (target) {
                        dcSum += dcMult * target(x, y, t)[c];
                    } else { 
                        // try to read the dc term out of the double
                        // integral of the gradient fields
                        // instead. Works if the gradients were
                        // computed with a zero boundary condition.
                        dcSum += 2.0f*((dx.width-x)*dx(x, y, t)[c] + (dy.height-y)*dy(x, y, t)[c]);
                    }
                    
                    
                    if (target) 
                        *fftPtr = targetStrength * target(x, y, t)[c];                    
                    else 
                        *fftPtr = 0;

                    // Subtract g^x_x and g^y_y, with boundary factor of -2.0 to account for boundary reflections implicit in the DCT
                    if (x == 0) {
                        *fftPtr -= (+2.0f * dx(x+1, y, t)[c]);
                    } else if (x == dx.width - 1) {
                        *fftPtr -= (-2.0f * dx(x, y, t)[c]);
                    } else {
                        *fftPtr -= (dx(x+1, y, t)[c] - dx(x, y, t)[c]);
                    }

                    if (y == 0) {
                        *fftPtr -= (+2.0f * dy(x, y+1, t)[c]);
                    } else if (y == dx.height -1) {
                        *fftPtr -= (-2.0f * dy(x, y, t)[c]);
                    } else {
                        *fftPtr -= (dy(x, y+1, t)[c] - dy(x, y, t)[c]);
                    }

                    
                    fftPtr++;
                }
            }

            //transform h_hat to H_hat by taking the DCT of h_hat
            fftwf_execute(fftPlan);

            //compute F_hat using H_hat (see equation 29 in the paper)
            fftPtr = fftBuff(0, 0);
            for(int y = 0; y < dx.height; y++) {
                for(int x = 0; x < dx.width; x++) {
                    float ftLapResponse = ftLapY(0, y)[0] + ftLapX(x, 0)[0];
                    *fftPtr++ /= (targetStrength - ftLapResponse);
                }
            }
            
            fftBuff(0, 0)[0] = dcSum;
            
            //transform F_hat to f_hat by taking the inverse DCT of F_hat
            fftwf_execute(fftPlan);

            float fftMult = 1.0f / (4.0f * (dx.width-1) * (dx.height-1));

            fftPtr = fftBuff(0, 0);
            
            for(int y = 0; y < dx.height; y++) {
                for(int x = 0; x < dx.width; x++) {
                    out(x, y, t)[c] = (*fftPtr++) * fftMult;
                }
            }
            
        }
    }

    fftwf_destroy_plan(fftPlan);

    return out;
    
}


#include "footer.h"
#endif

