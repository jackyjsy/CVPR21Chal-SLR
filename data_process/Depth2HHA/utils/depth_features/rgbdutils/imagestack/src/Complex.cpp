#include "main.h"
#include "Complex.h"
#include "header.h"

void ComplexMultiply::help() {
    pprintf("-complexmultiply multiplies the top image in the stack by the second"
            " image in the stack, using 2 \"complex\" images as its input - a"
            " \"complex\" image is one where channel 2*n is the real part of the nth"
            " channel and channel 2*n + 1 is the imaginary part of the nth"
            " channel. Using zero arguments results in a straight multiplication"
            " (a + bi) * (c + di), using one argument results in a conjugate"
            " multiplication (a - bi) * (c + di).\n"
            "\n"
            "Usage: ImageStack -load a.tga -load b.tga -complexmultiply -save out.tga.\n");
}

void ComplexMultiply::parse(vector<string> args) {
    assert(args.size() < 2, "-complexmultiply takes zero or one arguments\n");
    if (stack(0).channels == 2 && stack(1).channels > 2) {
      apply(stack(1), stack(0), (bool)args.size());
      pop();
    } else {
      apply(stack(0), stack(1), (bool)args.size());
      pull(1);
      pop();
    }
}

void ComplexMultiply::apply(Window a, Window b, bool conj = false) {
    assert(a.channels % 2 == 0 && b.channels % 2 == 0,
           "-complexmultiply requires images with an even number of channels (%d %d)\n", 
           a.channels, b.channels);

    assert(a.frames == b.frames &&
           a.width == b.width &&
           a.height == b.height,
           "images must be the same size\n");

    float a1, b1, a2, b2;
    int sign = (conj)? -1 : 1;
    if (a.channels != 2 && b.channels == 2) {
        for (int t = 0; t < a.frames; t++) {
            for (int y = 0; y < a.height; y++) {
                for (int x = 0; x < a.width; x++) {
                    for (int c = 0; c < a.channels; c+=2) {
                        a1 = a(x, y, t)[c];
                        b1 = a(x, y, t)[c+1];
                        a2 = b(x, y, t)[0];
                        b2 = b(x, y, t)[1];
                        a(x, y, t)[c] = a1*a2 - sign*b1*b2;
                        a(x, y, t)[c+1] = b1*a2 + sign*b2*a1;
                    }
                }
            }
        }
    } else {
        for (int t = 0; t < a.frames; t++) {
            for (int y = 0; y < a.height; y++) {
                for (int x = 0; x < a.width; x++) {
                    for (int c = 0; c < a.channels; c+=2) {
                        a1 = a(x, y, t)[c];
                        b1 = a(x, y, t)[c+1];
                        a2 = b(x, y, t)[c];
                        b2 = b(x, y, t)[c+1];
                        a(x, y, t)[c] = a1*a2 - sign*b1*b2;
                        a(x, y, t)[c+1] = b1*a2 + sign*b2*a1;
                    }
                }
            }
        }
    }
}


void ComplexDivide::help() {
    pprintf("-complexdivide divides the top image in the stack by the second image"
            " in the stack, using 2 \"complex\" images as its input - a \"complex\""
            " image is one where channel 2*n is the real part of the nth channel and"
            " channel 2*n + 1 is the imaginary part of the nth channel. Using zero"
            " arguments results in a straight division (a + bi) / (c + di). Using"
            " one argument results in a conjugate division (a - bi) / (c + di).\n"
            "\n"
            "Usage: ImageStack -load a.tga -load b.tga -complexdivide -save out.tga.\n");
}

void ComplexDivide::parse(vector<string> args) {
    assert(args.size() == 0 || args.size() == 1, "-complexdivide takes zero or one arguments\n");
    if (stack(0).channels == 2 && stack(1).channels > 2) {
        apply(stack(1), stack(0), (bool)args.size());
        pop();
    } else {
        apply(stack(0), stack(1), (bool)args.size());
        pull(1);
        pop();
    }
}

void ComplexDivide::apply(Window a, Window b, bool conj = false) {
    assert(a.channels % 2 == 0 && b.channels % 2 == 0,
           "-complexdivide requires images with an even number of channels\n");

    assert(a.frames == b.frames &&
           a.width == b.width &&
           a.height == b.height,
           "images must be the same size\n");

    float a1, b1, a2, b2, denom;
    int sign = (conj)? -1 : 1;
    if (a.channels != 2 && b.channels == 2) {
        for (int t = 0; t < a.frames; t++) {
            for (int y = 0; y < a.height; y++) {
                for (int x = 0; x < a.width; x++) {
                    for (int c = 0; c < a.channels; c+=2) {
                        a1 = a(x, y, t)[c];
                        b1 = a(x, y, t)[c+1];
                        a2 = b(x, y, t)[0];
                        b2 = b(x, y, t)[1];
                        denom = a2*a2 + b2*b2;
                        a(x, y, t)[c] = (a1*a2 + sign*b1*b2)/denom;
                        a(x, y, t)[c+1] = (sign*b1*a2 - b2*a1)/denom;
                    }
                }
            }
        }
    } else {
        for (int t = 0; t < a.frames; t++) {
            for (int y = 0; y < a.height; y++) {
                for (int x = 0; x < a.width; x++) {
                    for (int c = 0; c < a.channels; c+=2) {
                        a1 = a(x, y, t)[c];
                        b1 = a(x, y, t)[c+1];
                        a2 = b(x, y, t)[c];
                        b2 = b(x, y, t)[c+1];
                        denom = a2*a2 + b2*b2;
                        a(x, y, t)[c] = (a1*a2 + sign*b1*b2)/denom;
                        a(x, y, t)[c+1] = (sign*b1*a2 - b2*a1)/denom;
                    }
                }
            }
        }
    }
}


void ComplexReal::help() {
    pprintf("-complexreal takes a \"complex\" image, in which the even channels"
            " represent the real component and the odd channels represent the"
            " imaginary component, and produces an image containing only the real"
            " channels.\n"
            "\n"
            "Usage: ImageStack -load a.png -fftreal -complexreal -display\n");
}

void ComplexReal::parse(vector<string> args) {   
    assert(args.size() == 0, "-complexreal takes no arguments\n");
    Image im = apply(stack(0));
    pop();
    push(im);
}

Image ComplexReal::apply(Window im) {
    assert(im.channels % 2 == 0, "complex images must have an even number of channels\n");

    Image out(im.width, im.height, im.frames, im.channels/2);

    float *outPtr = out(0, 0, 0);
    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            float *imPtr = im(0, y, t);
            for (int x = 0; x < im.width; x++) { 
                for (int c = 0; c < im.channels; c+=2) {
                    *outPtr++ = *imPtr++;
                    imPtr++;
                }
            }
        }
    }

    return out;
}

void RealComplex::help() {
    pprintf("-realcomplex takes a \"real\" image, and converts it to a \"complex\""
            " image, in which the even channels represent the real component and"
            " the odd channels represent the imaginary component.\n"
            "\n"
            "Usage: ImageStack -load a.png -realcomplex -fft -display\n");
}

void RealComplex::parse(vector<string> args) {   
    assert(args.size() == 0, "-complexreal takes no arguments\n");
    Image im = apply(stack(0));
    pop();
    push(im);
}

Image RealComplex::apply(Window im) {
    Image out(im.width, im.height, im.frames, im.channels*2);

    float *outPtr = out(0, 0, 0);
    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            float *imPtr = im(0, y, t);
            for (int x = 0; x < im.width; x++) { 
                for (int c = 0; c < im.channels; c++) {
                    *outPtr++ = *imPtr++;
                    *outPtr++ = 0;
                }
            }
        }
    }

    return out;
}

void ComplexImag::help() {
    pprintf("-compleximag takes a \"complex\" image, in which the even channels"
            " represent the real component and the odd channels represent the"
            " imaginary component, and produces an image containing only the imaginary"
            " channels.\n"
            "\n"
            "Usage: ImageStack -load a.png -fftreal -compleximag -display\n");
}

void ComplexImag::parse(vector<string> args) {   
    assert(args.size() == 0, "-compleximag takes no arguments\n");
    Image im = apply(stack(0));
    pop();
    push(im);
}

Image ComplexImag::apply(Window im) {
    assert(im.channels % 2 == 0, "complex images must have an even number of channels\n");

    Image out(im.width, im.height, im.frames, im.channels/2);

    float *outPtr = out(0, 0, 0);
    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            float *imPtr = im(0, y, t);
            for (int x = 0; x < im.width; x++) { 
                for (int c = 0; c < im.channels; c+=2) {
                    imPtr++;
                    *outPtr++ = *imPtr++;
                }
            }
        }
    }

    return out;
}


void ComplexMagnitude::help() {
    pprintf("-complexmagnitude takes a \"complex\" image, in which the even channels"
            " represent the real component and the odd channels represent the"
            " imaginary component, and produces an image containing the complex"
            " magnitude\n"
            "\n"
            "Usage: ImageStack -load a.png -fftreal -complexmagnitude -display\n");
}

void ComplexMagnitude::parse(vector<string> args) {   
    assert(args.size() == 0, "-complexmagnitude takes no arguments\n");
    Image im = apply(stack(0));
    pop();
    push(im);
}

Image ComplexMagnitude::apply(Window im) {
    assert(im.channels % 2 == 0, "complex images must have an even number of channels\n");

    Image out(im.width, im.height, im.frames, im.channels/2);

    float *outPtr = out(0, 0, 0);
    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            float *imPtr = im(0, y, t);
            for (int x = 0; x < im.width; x++) { 
                for (int c = 0; c < im.channels; c+=2) {
                    float real = *imPtr++;
                    float imag = *imPtr++;
                    *outPtr++ = sqrtf(real*real + imag*imag);
                }
            }
        }
    }

    return out;
}



void ComplexPhase::help() {
    pprintf("-complexphase takes a \"complex\" image, in which the even channels"
            " represent the real component and the odd channels represent the"
            " imaginary component, and produces an image containing the complex"
            " phase\n"
            "\n"
            "Usage: ImageStack -load a.png -fftreal -complexphase -display\n");
}

void ComplexPhase::parse(vector<string> args) {   
    assert(args.size() == 0, "-complexphase takes no arguments\n");
    Image im = apply(stack(0));
    pop();
    push(im);
}

Image ComplexPhase::apply(Window im) {
    assert(im.channels % 2 == 0, "complex images must have an even number of channels\n");

    Image out(im.width, im.height, im.frames, im.channels/2);

    float *outPtr = out(0, 0, 0);
    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            float *imPtr = im(0, y, t);
            for (int x = 0; x < im.width; x++) { 
                for (int c = 0; c < im.channels; c+=2) {
                    float real = *imPtr++;
                    float imag = *imPtr++;
                    *outPtr++ = atan2(imag, real);
                }
            }
        }
    }

    return out;
}


void ComplexConjugate::help() {
    pprintf("-complexconjugate takes a \"complex\" image, in which the even channels"
            " represent the real component and the odd channels represent the"
            " imaginary component, and produces an image containing the complex"
            " conjugate\n"
            "\n"
            "Usage: ImageStack -load a.png -fftreal -complexconjugate -display\n");
}

void ComplexConjugate::parse(vector<string> args) {   
    assert(args.size() == 0, "-complexconjugate takes no arguments\n");
    apply(stack(0));
}

void ComplexConjugate::apply(Window im) {
    assert(im.channels % 2 == 0, "complex images must have an even number of channels\n");

    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            float *imPtr = im(0, y, t);
            for (int x = 0; x < im.width; x++) { 
                for (int c = 0; c < im.channels; c+=2) {
                    imPtr++;
                    imPtr[0] = -imPtr[0];
                    imPtr++;
                }
            }
        }
    }
}

#include "footer.h"
