#include "main.h"
#include "Convolve.h"
#include "Arithmetic.h"
#include "Geometry.h"
#include "DFT.h"
#include "File.h"
#include "header.h"

void Convolve::help() {
    pprintf("-convolve takes a width, height, and frames and a single-channel 3D"
            " kernel specified across the rows, then down the columns, then over"
            " time, and convolves the current image by that matrix independently in"
            " each channel.\n"
            "\n"
            "With no numeric arguments, -convolve will use the next image on the stack as"
            " the filter.\n"
            "\n"
            "Boundary conditions can be specified by appending the argument"
            " \"zero\", \"homogeneous\", \"clamp\", or \"wrap\", which result in the"
            " respective assumptions: the image is zero outside the boundary; the"
            " image is weighted with weight one inside the boundary, and weight zero"
            " outside the boundary; the image values outside the boundary clamp to"
            " the nearest defined image value (Neumann); and the image wraps outside"
            " the boundary.\n"
            "\n"
            "Convolution by multi-channel filters is poorly defined, because it"
            " requires a vector-vector multiplication between filter values and"
            " image values. By specifying a final argument of \"inner\", \"outer\","
            " or \"elementwise\", the multiplication used is correspondingly the"
            " inner product (or matrix product if the image and kernel have a"
            " differing number of frames); the outer product; or an elementwise"
            " product. If the kernel has k channels and the image has m channels,"
            " \"inner\" produces an image with max(m/k, k/m) channels, \"outer\""
            " produces an image with m*k channels, and \"elementwise\" requires"
            " that m==k and produces an image with the same number of channels. The"
            " default method is \"outer\".\n"
            "\n"
            "Taking a horizontal gradient with zero boundary condition: \n"
            " ImageStack -load a.tga -convolve 2 1 1  -1 1 zero -save dx.tga\n"
            "Convolving by a bank of filters: \n"
            " ImageStack -load bank.tmp -load a.tga -convolve homogeneous outer\n");
}

void Convolve::parse(vector<string> args) {
    string boundaryCondition = "homogeneous";
    string channelMode = "outer";

    Image filter;

    if (args.size() > 3) {
        int frames, width, height;
        size_t size;
        width = readInt(args[0]);
        height = readInt(args[1]);
        frames = readInt(args[2]);
        size = frames * width * height;
        assert(args.size() >= size+3,
               "a size of %ix%ix%i requires at least %i more arguments. %i were given.", 
               width, height, frames, size, (int)args.size() - 3);
        assert(size % 2 == 1, "filter must have odd size\n");

        assert(args.size() <= size+4,
               "a size of %ix%ix%i requires at most %i more arguments. %i were given.", 
               width, height, frames, size, (int)args.size() - 2);
        
        filter = Image(width, height, frames, 1);

        size_t i = 3;
        float *filterPtr = filter(0, 0, 0);
        for (int t = 0; t < frames; t++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    *filterPtr++ =  readFloat(args[i]);
                    i++;
                }
            }
        }

        if (args.size() == size+4) {
            boundaryCondition = args[size+3];
        }

    } else if (args.size() < 3) {
        filter = stack(1);
        if (args.size() >= 1) {
            boundaryCondition = args[0];
        }
        if (args.size() == 2) {
            channelMode = args[1];
        }
    } else {
        panic("-convolve needs either zero, one, two, or at least four arguments\n");
    }

    Multiply::Mode m;
    BoundaryCondition b;

    if (boundaryCondition == "zero") b = Zero;
    else if (boundaryCondition == "homogeneous") b = Homogeneous;
    else if (boundaryCondition == "clamp") b = Clamp;
    else if (boundaryCondition == "wrap") b = Wrap;
    else {
        panic("Unknown boundary condition: %s\n", boundaryCondition.c_str());
    }

    if (channelMode == "inner") m = Multiply::Inner;
    else if (channelMode == "outer") m = Multiply::Outer;
    else if (channelMode == "elementwise") m = Multiply::Elementwise;
    else {
        panic("Unknown vector-vector multiplication: %s\n", channelMode.c_str());
    }
        
    Image im = apply(stack(0), filter, b, m);
    pop();
    push(im);

}

// all the various possible vector-vector multiplications used in
// convolution. A pointer into the kernel is argument a, and a pointer
// into the image is argument b.

// Each kernel entry is a matrix to be applied to each image entry
inline void Convolve__Inner1(float *a, int na, float *b, int nb, float *out, int nout) {
    // int nout = na/nb;   
    for (int i = 0; i < nout; i++) {
        for (int j = 0; j < nb; j++) {
            *out += *(a++) * b[j];
        }
        out++;
    }
}

// Each image entry is a matrix to be applied to each kernel entry
inline void Convolve__Inner2(float *a, int na, float *b, int nb, float *out, int nout) {
    Convolve__Inner1(b, nb, a, na, out, nout);
}

// Each kernel entry is a bank of filters to be applied to each
// channel independently
inline void Convolve__Outer(float *a, int na, float *b, int nb, float *out, int nout) {
    for (int i = 0; i < na; i++) {
        for (int j = 0; j < nb; j++) {
            (*out++) += a[i] * b[j];
        }
    }
}

// Special case of the above when there is one filter in the
// bank. This is the most common type of convolution.
inline void Convolve__Outer1(float *a, int na, float *b, int nb, float *out, int nout) {
    for (int j = 0; j < nb; j++) {
        (*out++) += (*a) * b[j];
    }
}

// Each kernel entry should be multiplied element-wise by each image entry
inline void Convolve__Elementwise(float *a, int na, float *b, int nb, float *out, int nout) {
    for (int j = 0; j < na; j++) {
        (*out++) += (*a++) * (*b++);
    }
}

// Both are single-channel
inline void Convolve__Scalar(float *a, int na, float *b, int nb, float *out, int nout) {
    out[0] += a[0] * b[0];
}

// Make a more convenient name to refer to the various types of vector-vector multiplication
typedef void (*Convolve__VectorVectorMult)(float *, int, float *, int, float *, int);

template<Convolve::BoundaryCondition b, Convolve__VectorVectorMult m>
static Image Convolve__apply(Window im, Window filter, Image out) {

    int filterSize = filter.frames * filter.width * filter.height;
    assert(filterSize % 2 == 1, "filter must have odd size\n");

    int xoff = (filter.width - 1)/2;
    int yoff = (filter.height - 1)/2;
    int toff = (filter.frames - 1)/2;

    // Used by the homogeneous boundary condition
    vector<float> ones(im.channels, 1.0f);
    vector<float> filterSum(out.channels, 0.0f);
    if (b == Convolve::Homogeneous) {
        // compute the sum of weights in the non-boundary case
        for (int t = 0; t < filter.frames; t++) {
            for (int y = 0; y < filter.height; y++) {
                for (int x = 0; x < filter.width; x++) {
                    m(filter(x, y, t), filter.channels,
                      &ones[0], im.channels,
                      &filterSum[0], out.channels);
                }
            }
        }

        for (int c = 0; c < out.channels; c++) {
            filterSum[c] = 1.0f/filterSum[c];
        }
    }
    vector<float> weight(out.channels, 0.0f);
    
    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                float *outPtr = out(x, y, t);

                bool boundary = (x - xoff < 0 ||
                                 x + xoff > im.width-1 ||
                                 y - yoff < 0 ||
                                 y + yoff > im.height-1 ||
                                 t - toff < 0 ||
                                 t + toff > im.frames-1);

                if (!boundary) {
                    for (int dt = -toff; dt <= toff; dt++) {
                        for (int dy = -yoff; dy <= yoff; dy++) {
                            float *filterPtr = filter(filter.width-1, -dy+yoff, -dt+toff);
                            float *imPtr = im(x-xoff, y+dy, t+dt);
                            for (int dx = -xoff; dx <= xoff; dx++) {
                                 m(filterPtr, filter.channels,
                                  imPtr, im.channels,
                                  outPtr, out.channels);                                
                                filterPtr -= filter.channels;
                                imPtr += im.channels;
                            }
                        }
                    }                    

                    if (b == Convolve::Homogeneous) {
                        // Renormalize the output to have the right sum of weights
                        for (int c = 0; c < out.channels; c++) {
                            outPtr[c] *= filterSum[c];
                        } 
                    }
                } else if (b == Convolve::Zero || b == Convolve::Homogeneous) {
                    if (b == Convolve::Homogeneous) {
                        for (int c = 0; c < out.channels; c++) {
                            weight[c] = 0.0f;
                        }
                    }


                    for (int dt = -toff; dt <= toff; dt++) {
                        if (t + dt < 0) continue;
                        if (t + dt > im.frames-1) break;
                        for (int dy = -yoff; dy <= yoff; dy++) {
                            if (y + dy < 0) continue;
                            if (y + dy > im.height-1) break;
                            float *filterPtr = filter(filter.width-1, -dy+yoff, -dt+toff);
                            float *imPtr = im(x-xoff, y+dy, t+dt);
                            for (int dx = -xoff; dx <= xoff; dx++) {
                                if (x + dx < 0) {
                                    imPtr += im.channels;
                                    filterPtr -= filter.channels;
                                    continue;
                                }
                                if (x + dx > im.width-1) break;
                                
                                m(filterPtr, filter.channels,
                                  imPtr, im.channels,
                                  outPtr, out.channels);
                                if (b == Convolve::Homogeneous) {
                                    // What would have been the effect if
                                    // I did the same thing for all-ones
                                    m(filterPtr, filter.channels,
                                      &ones[0], im.channels,
                                      &weight[0], out.channels);                                    
                                }

                                filterPtr -= filter.channels;
                                imPtr += im.channels;
                            }
                        }
                    }

                    if (b == Convolve::Homogeneous) {
                        // Renormalize the output to have the right sum of weights
                        for (int c = 0; c < out.channels; c++) {
                            outPtr[c] /= weight[c];
                        } 
                    }
                } else if (b == Convolve::Clamp) {
                    for (int dt = -toff; dt <= toff; dt++) {
                        int imt = clamp(t + dt, 0, im.frames-1);
                        for (int dy = -yoff; dy <= yoff; dy++) {
                            int imy = clamp(y + dy, 0, im.height-1);
                            float *filterPtr = filter(filter.width-1, -dy+yoff, -dt+toff);
                            float *imPtr = im(0, imy, imt);
                            for (int dx = -xoff; dx <= xoff; dx++) {
                                int imx = clamp(x + dx, 0, im.width-1);
                                m(filterPtr, filter.channels,
                                  imPtr + imx * im.xstride, im.channels,
                                  outPtr, out.channels);
                                filterPtr -= filter.channels;
                            }
                        }
                    }
                } else if (b == Convolve::Wrap) {
                    for (int dt = -toff; dt <= toff; dt++) {
                        int imt = t + dt;
                        while (imt < 0) imt += im.frames;
                        while (imt >= im.frames) imt -= im.frames;
                        for (int dy = -yoff; dy <= yoff; dy++) {
                            int imy = y + dy;
                            while (imy < 0) imy += im.height;
                            while (imy >= im.height) imy -= im.height;
                            float *filterPtr = filter(filter.width-1, -dy+yoff, -dt+toff);
                            float *imPtr = im(0, imy, imt);
                            for (int dx = -xoff; dx <= xoff; dx++) {
                                int imx = x + dx;
                                while (imx < 0) imx += im.width;
                                while (imx >= im.width) imx -= im.width;
                                m(filterPtr, filter.channels,
                                  imPtr + imx * im.xstride, im.channels,
                                  outPtr, out.channels);
                                filterPtr -= filter.channels;
                            }
                        }
                    }
                } else {
                    panic("Unknown boundary condition: %d\n", b);
                }
            }
        }
    }

    return out;
}


// This function is a jumping off point for the fully-templatized version above
template<Convolve__VectorVectorMult m>
Image Convolve__apply(Window im, Window filter, Image out, Convolve::BoundaryCondition b) {   
    switch(b) {
        case Convolve::Zero:
            return Convolve__apply<Convolve::Zero, m>(im, filter, out);
        case Convolve::Homogeneous:
            return Convolve__apply<Convolve::Homogeneous, m>(im, filter, out);
        case Convolve::Clamp:
            return Convolve__apply<Convolve::Clamp, m>(im, filter, out);
        case Convolve::Wrap:
            return Convolve__apply<Convolve::Wrap, m>(im, filter, out);
    }
    panic("Unknown boundary condition: %d\n", b);
    return Image();
}

Image Convolve::apply(Window im, Window filter, BoundaryCondition b, Multiply::Mode m) {
    // This function is a jumping off point for the partially-templatized version above

    #ifndef NO_FFTW
    if (filter.width * filter.height * filter.frames > 50) {
        return FFTConvolve::apply(im, filter, b, m);
    }
    #endif
    
    if (im.channels == 1 && filter.channels == 1) {
        // INNER, OUTER, and ELEMENTWISE all have the same meaning here
        Image out(im.width, im.height, im.frames, 1);
        return Convolve__apply<Convolve__Scalar>(im, filter, out, b);
    }

    if (m == Multiply::Inner) {
        if (im.channels < filter.channels && filter.channels % im.channels == 0) {
            Image out(im.width, im.height, im.frames, filter.channels / im.channels);
            return Convolve__apply<Convolve__Inner1>(im, filter, out, b);
        } else if (im.channels >= filter.channels && im.channels % filter.channels == 0) {
            Image out(im.width, im.height, im.frames, im.channels / filter.channels);
            return Convolve__apply<Convolve__Inner2>(im, filter, out, b);
        } else {
            panic("For inner products, either the number of channels in the filter must"
                  "be a multiple of the number of channels in the image, or vice-versa\n");
        }
    } else if (m == Multiply::Outer) {
        Image out(im.width, im.height, im.frames, filter.channels * im.channels);
        if (filter.channels == 1) { // common case optimization
            Convolve__apply<Convolve__Outer1>(im, filter, out, b);
        } else {
            Convolve__apply<Convolve__Outer>(im, filter, out, b);
        }
        return out;
    } else if (m == Multiply::Elementwise) {
        Image out(im.width, im.height, im.frames, im.channels);
        if (im.channels != filter.channels) {
            panic("For elementwise multiplication, the filter must have the same"
                  " number of channels as the image\n");
        }        
        return Convolve__apply<Convolve__Elementwise>(im, filter, out, b);
    }

    panic("Unknown channel mode: %d\n", m);
    return Image();
}




void Deconvolve::help() {
    // TODO
}

void Deconvolve::parse(vector<string> args) {
    if (args.size() > 4) {
        int frames, width, height, size;
        float seconds = readFloat(args[0]);
        frames = readInt(args[1]);
        width = readInt(args[2]);
        height = readInt(args[3]);
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
        
        
        Image im = apply(stack(0), filter, seconds);
        pop();
        push(im);
    } else if (args.size() == 1) {
        assert(stack(1).channels == 1, "filter must be single channel\n");
        float seconds = readFloat(args[0]);
        Image im = apply(stack(0), stack(1), seconds);
        pop();
        pop();
        push(im);
    } else {
        panic("-deconvolve needs either one or at least five arguments\n");
    }    
}

Image Deconvolve::apply(Window im, Window filter, float maxTime, int maxIterations) {
    return apply(im, filter, im, maxTime, maxIterations);
}

Image Deconvolve::apply(Window im, Window filter, Window initialGuess, float maxTime, int maxIterations) {
    float finish = currentTime() + maxTime;

    Image target(im);

    // figure out some properties of the kernel
    bool symmetric = true;
    for (int t = 0; t < filter.frames && symmetric; t++) {
        for (int y = 0; y < filter.height && symmetric; y++) {
            for (int x = 0; x < (filter.width+1)/2 && symmetric; x++) {
                float val1 = filter(x, y, t)[0];
                float val2 = filter(filter.width-1-x, filter.height-1-y, filter.frames-1-t)[0];
                if (fabs(val1 - val2) > 1e-20) symmetric = false;
            }
        }
    }

    bool positiveDefinite = false;
    // In general, filters are not symmetric positive-definite, unless their Fourier transform is all positive
    #ifndef NO_FFTW
    {

        Image fftFilter(im.width, im.height, im.frames, filter.channels*2);
        // cyclic rotate and add a complex channel
        for (int t = 0; t < filter.frames; t++) {
            int ft = t - filter.frames/2;
            if (ft < 0) ft += im.frames;
            for (int y = 0; y < filter.height; y++) {
                int fy = y - filter.height/2;
                if (fy < 0) fy += im.height;
                for (int x = 0; x < filter.width; x++) {
                    int fx = x - filter.width/2;
                    if (fx < 0) fx += im.width;
                    for (int c = 0; c < filter.channels; c++) {
                        fftFilter(fx, fy, ft)[2*c] = filter(x, y, t)[c];
                    }
                }
            }
        }

        FFT::apply(fftFilter);

        positiveDefinite = true;
        float *fPtr = fftFilter(0, 0, 0);
        for (int t = 0; t < fftFilter.frames; t++) {
            for (int y = 0; y < fftFilter.height; y++) {
                for (int x = 0; x < fftFilter.width; x++) {
                    for (int c = 0; c < filter.channels; c++) {
                        if (*fPtr++ < -0.00001) positiveDefinite = false;
                        if (*fPtr > 0.00001) positiveDefinite = false;
                        if (*fPtr++ < -0.00001) positiveDefinite = false;
                    }
                }
            }
        }
    }
    #endif    

    printf("Positive definite: %s\n", positiveDefinite ? "true" : "false");
    printf("Symmetric: %s\n", symmetric ? "true" : "false");

    // if the kernel is not symmetric positive-definite, we need to convolve by its transpose too
    Image filterTranspose(filter);
    Flip::apply(filterTranspose, 'x');
    Flip::apply(filterTranspose, 'y');
    Flip::apply(filterTranspose, 't');

    if (!symmetric || !positiveDefinite) {
        printf("Convolving with filter transpose.\n");
        target = Convolve::apply(im, filterTranspose, Convolve::Clamp);
    }

    // Given that we don't know what the boundary condition was that
    // created this blurry image, we down-weight the boundaries in the least squares solve
    float filterSum = 0;
    for (int t = 0; t < filter.frames; t++) {
        for (int y = 0; y < filter.height; y++) {
            for (int x = 0; x < filter.width; x++) {
                filterSum += filter(x, y, t)[0];
            }
        }
    }

    printf("Starting CG for filter size %i %i %i\n", filter.width, filter.height, filter.frames);

    Image x(initialGuess);

    // r = b - Ax
    Image r = target.copy();
    Image q = Convolve::apply(x, filter, Convolve::Clamp); 
    if (!symmetric || !positiveDefinite) q = Convolve::apply(q, filterTranspose, Convolve::Clamp);
    //Multiply::apply(q, weight);
    Subtract::apply(r, q);

    // d = r
    Image d = r.copy();       

    double errorNew = norm(r);
    double errorOld = errorNew;

    for (int i = 0; i < maxIterations || maxIterations <= 0; i++) {

        // q = Ad
        q = Convolve::apply(d, filter, Convolve::Clamp);
        if (!symmetric || !positiveDefinite) q = Convolve::apply(q, filterTranspose, Convolve::Clamp);
        //Multiply::apply(q, weight);

        float alpha = errorNew / dot(d, q);

        // x = alpha*d + x;
        scaleAdd(x, alpha, d, x);

        if (i % 50 == 49) {
            // remove accumulated error in the residual
            // r = b - Ax
            r = target.copy();
            q = Convolve::apply(x, filter, Convolve::Clamp);
            if (!symmetric || !positiveDefinite) q = Convolve::apply(q, filterTranspose, Convolve::Clamp);
            //Multiply::apply(q, weight);
            Subtract::apply(r, q);
        } else {
            // estimate new residual directly
            // r = -alpha*q + r
            scaleAdd(r, -alpha, q, r);
        }        
        
        errorOld = errorNew;
        errorNew = norm(r);

        // d = beta*d + r
        double beta = errorNew / errorOld;
        scaleAdd(d, beta, d, r);

        if (maxTime > 0) {
            printf("%d: Error: %3.10f  \t (%3.1f seconds to go)\n", 
                   i, errorNew/(target.width*target.height*target.frames), finish - currentTime());
        } else {
            printf("%d: Error: %3.10f\n",
                   i, errorNew/(target.width*target.height*target.frames));
        }

        if (errorNew < 0.00000000001 || currentTime() > finish) break;
    }    
    
    return x;
}



double Deconvolve::dot(Image &a, Image &b) {
    double val = 0;
    float *ptr1 = a(0, 0, 0), *ptr2 = b(0, 0, 0);
    int size = a.frames * a.width * a.height * a.channels;
    for (int i = 0; i < size; i++) {
        val += (*ptr1++) * (*ptr2++);
    }
    return val;
}

double Deconvolve::norm(Image &a) {
    double val = 0;
    float *ptr = a(0, 0, 0);
    int size = a.frames * a.width * a.height * a.channels;
    for (int i = 0; i < size; i++) {
        val += (*ptr) * (*ptr);
        ptr++;
    }
    return val;
}

// compute a = alpha*b + c
void Deconvolve::scaleAdd(Image a, float alpha, Image &b, Image &c) {
    float *aPtr = a(0, 0, 0);
    float *bPtr = b(0, 0, 0);
    float *cPtr = c(0, 0, 0);
    int size = a.frames * a.width * a.height * a.channels;
    for (int i = 0; i < size; i++) {
        *aPtr++ = alpha * (*bPtr++) + (*cPtr++);
    }        
}

#include "footer.h"

