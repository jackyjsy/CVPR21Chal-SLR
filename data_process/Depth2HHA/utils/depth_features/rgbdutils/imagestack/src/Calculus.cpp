#include "main.h"
#include "Calculus.h"
#include "Geometry.h"
#include "Arithmetic.h"
#include "Stack.h"
#include "Convolve.h"
#include "Filter.h"
#include "File.h"
#include "Display.h"
#include "LAHBPCG.h"
#include "header.h"

void Gradient::help() {
    printf("\n-gradient takes the backward differences in the dimension specified by the\n"
           "argument. Values outside the image are assumed to be zero, so the first row,\n"
           "or column, or frame, will not change, effectively storing the initial value\n"
           "to make later integration easy. Multiple arguments can be given to differentiate\n"
           "with respect to multiple dimensions in order (although the order does not matter).\n\n"
           "Warning: Don't expect to differentiate more than twice and be able to get back\n"
           "the image by integrating. Numerical errors will dominate.\n\n"
           "Usage: ImageStack -load a.tga -gradient x y -save out.tga\n\n");
}

void Gradient::parse(vector<string> args) {
    assert(args.size() > 0, "-gradient requires at least one argument\n");
    for (size_t i = 0; i < args.size(); i++) {
        apply(stack(0), args[i]);
    }
}

// gradient can be called as gradient('t') or gradient("xyt") 
void Gradient::apply(Window im, string dimensions) {
    for (size_t i = 0; i < dimensions.size(); i++) {
        apply(im, dimensions[i]);
    }
}

void Gradient::apply(Window im, char dimension) {
    int mint = 0, minx = 0, miny = 0;
    int dt = 0, dx = 0, dy = 0;

    if (dimension == 'x') {
        dx = 1;
        minx = 1;
    } else if (dimension == 'y') {
        dy = 1;
        miny = 1;
    } else if (dimension == 't') {
        dt = 1;
        mint = 1;
    } else {
        panic("Must differentiate with respect to x, y, or t\n");
    }

    // walk backwards through the data, looking at the untouched data for the differences
    for (int t = im.frames - 1; t >= mint; t--) {
        for (int y = im.height - 1; y >= miny; y--) {
            for (int x = im.width - 1; x >= minx; x--) {
                for (int c = 0; c < im.channels; c++) {
                    im(x, y, t)[c] -= im(x - dx, y - dy, t - dt)[c];
                }
            }
        }
    }
}


void Integrate::help() {
    printf("\n-integrate computes partial sums along the given dimension. It is the\n"
           "of the -gradient operator. Multiply dimensions can be given as arguments,\n"
           "for example -integrate x y will produce a summed area table of an image.\n"
           "Allowed dimensions are x, y, or t.\n\n"
           "Warning: Don't expect to integrate more than twice and be able to get back\n"
           "the image by differentiating. Numerical errors will dominate.\n\n"
           "Usage: ImageStack -load a.tga -gradient x y -integrate y x -save a.tga\n\n");
}

void Integrate::parse(vector<string> args) {
    assert(args.size() > 0, "-integrate requires at least one argument\n");
    for (size_t i = 0; i < args.size(); i++) {
        apply(stack(0), args[i]);
    }        
}

// integrate can be called as integrate('t') or integrate("xyt") 
void Integrate::apply(Window im, string dimensions) {
    for (size_t i = 0; i < dimensions.size(); i++) {
        apply(im, dimensions[i]);
    }        
}

void Integrate::apply(Window im, char dimension) {
    int minx = 0, miny = 0, mint = 0;
    int dx = 0, dy = 0, dt = 0;

    if (dimension == 'x') {
        dx = 1;
        minx = 1;
    } else if (dimension == 'y') {
        dy = 1;
        miny = 1;
    } else if (dimension == 't') {
        dt = 1;
        mint = 1;
    } else {
        panic("Must integrate with respect to x, y, or t\n");
    }

    // walk forwards through the data, adding up as we go
    for (int t = mint; t < im.frames; t++) {
        for (int y = miny; y < im.height; y++) {
            for (int x = minx; x < im.width; x++) {
                for (int c = 0; c < im.channels; c++) {
                    im(x, y, t)[c] += im(x - dx, y - dy, t - dt)[c];
                }
            }
        }
    }

}


void GradMag::help() {
    printf("-gradmag computes the square gradient magnitude at each pixel in x and\n"
           "y. Temporal gradients are ignored. The gradient is estimated using\n"
           "backward differences, and the image is assumed to be zero outside its\n"
           "bounds.\n\n"
           "Usage: ImageStack -load input.jpg -gradmag -save out.jpg\n");

}

void GradMag::parse(vector<string> args) {
    assert(args.size() == 0, "-laplacian takes no arguments\n");
    Image im = apply(stack(0));
    pop();
    push(im);
}

Image GradMag::apply(Window im) {
    Image out(im);
    Gradient::apply(im, 'x');
    Gradient::apply(out, 'y');

    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                for (int c = 0; c < im.channels; c++) {
                    out(x, y, t)[c] = im(x, y, t)[c] * im(x, y, t)[c] + out(x, y, t)[c] * out(x, y, t)[c];
                }
            }
        }
    }

    return out;
}


void Poisson::help() {
    pprintf("-poisson assumes the stack contains gradients images in x and y, and"
            " attempts to find the image which fits those gradients best in a least"
            " squares sense. It uses a preconditioned conjugate gradient descent"
            " method. It takes one argument, which is required RMS error of the"
            " result. This defaults to 0.01 if not given.\n"
            "\n"
            "Usage: ImageStack -load dx.tmp dy.tmp \n"
            "                  -poisson 0.0001 -save out.tga\n\n");
}

void Poisson::parse(vector<string> args) {
    assert(args.size() < 2, "-poisson requires one or fewer arguments\n");
    float rms = 0.01;
    if (args.size() > 0) {
        rms = readFloat(args[0]);
    }

    push(apply(stack(1), stack(0), rms));        
}

Image Poisson::apply(Window dx, Window dy, float rms) {
    assert(dx.width  == dy.width &&
           dx.height == dy.height && 
           dx.frames == dy.frames &&
           dx.channels == dy.channels,
           "derivatives must be matching size and number of channels\n");      

    
    Image zerosc(dx.width, dx.height, dx.frames, dx.channels);
    Image zeros1(dx.width, dx.height, dx.frames, 1);
    Image ones1(dx.width, dx.height, dx.frames, 1);
    Offset::apply(ones1, 1.0f);
    return LAHBPCG::apply(zerosc, dx, dy, zeros1, ones1, ones1, 999999, rms);
}

#include "footer.h"
