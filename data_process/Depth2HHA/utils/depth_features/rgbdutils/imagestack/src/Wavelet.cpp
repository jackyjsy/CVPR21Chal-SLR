#include "main.h"
#include "Wavelet.h"
#include "Geometry.h"
#include "header.h"

void Haar::help() {
    pprintf("-haar performs the standard 2D haar transform of an image. The image"
            " size must be a power of two. If given an integer argument k, it only"
            " recurses k times, and the image size must be a multiple of 2^k.\n"
            "\n"
            "Usage: ImageStack -load in.jpg -haar 1 -save out.jpg\n\n");
}

void Haar::parse(vector<string> args) {
    if (args.size() == 0) {
        apply(stack(0));
    } else if (args.size() == 1) {
        apply(stack(0), readInt(args[0]));
    } else {
        panic("-haar requires zero or one arguments\n");
    }
}

void Haar::apply(Window im, int times) {

    if (times <= 0) {
        assert(im.width == im.height, "to perform a full haar transorm, the image must be square.\n");
        times = 0;
        int w = im.width >> 1;
        while (w) {
            times++;
            w >>= 1;
        }
    }

    int factor = 1 << times;
    assert(im.width % factor == 0, "the image width is not a multiple of 2^%i", times);
    assert(im.height % factor == 0, "the image height is not a multiple of 2^%i", times);

    // transform in x
    Window win(im, 0, 0, 0, im.width, im.height, im.frames); 
    for (int i = 0; i < times; i++) {
        for (int t = 0; t < win.frames; t++) {            
            for (int y = 0; y < win.height; y++) {
                for (int x = 0; x < win.width; x+=2) {
                    float *a = win(x, y, t);
                    float *b = win(x+1, y, t);
                    for (int c = 0; c < win.channels; c++) {
                        float aVal = a[c];
                        float bVal = b[c];
                        a[c] = (aVal + bVal)/2;
                        b[c] = (bVal - aVal);
                    }
                }
            }
        }
        // separate into averages and differences
        Deinterleave::apply(win, 2, 1, 1);
        // repeat on the averages
        win = Window(win, 0, 0, 0, win.width/2, win.height, win.frames);
    }

    // transform in y
    win = Window(im, 0, 0, 0, im.width, im.height, im.frames); 
    for (int i = 0; i < times; i++) {
        for (int t = 0; t < win.frames; t++) {            
            for (int y = 0; y < win.height; y+=2) {
                for (int x = 0; x < win.width; x++) {
                    float *a = win(x, y, t);
                    float *b = win(x, y+1, t);
                    for (int c = 0; c < win.channels; c++) {
                        float aVal = a[c];
                        float bVal = b[c];
                        a[c] = (aVal + bVal)/2;
                        b[c] = (bVal - aVal);
                    }
                }
            }
        }
        // separate into averages and differences
        Deinterleave::apply(win, 1, 2, 1);
        // repeat on the averages
        win = Window(win, 0, 0, 0, win.width, win.height/2, win.frames);
    }
}


void InverseHaar::help() {
    printf("-inversehaar inverts the haar transformation with the same argument. See\n"
           "-help haar for detail.\n\n");

}

void InverseHaar::parse(vector<string> args) {
    if (args.size() == 0) {
        apply(stack(0));
    } else if (args.size() == 1) {
        apply(stack(0), readInt(args[0]));
    } else {
        panic("-haar requires zero or one arguments\n");
    }
}

void InverseHaar::apply(Window im, int times) {
    if (times <= 0) {
        assert(im.width == im.height, "to perform a full haar transorm, the image must be square.\n");
        times = 0;
        int w = im.width >> 1;
        while (w) {
            times++;
            w >>= 1;
        }
    }

    int factor = 1 << times;
    assert(im.width % factor == 0, "the image width is not a multiple of 2^%i", times);
    assert(im.height % factor == 0, "the image height is not a multiple of 2^%i", times);

    // transform in y
    int h = 2*im.height/factor;
    Window win(im, 0, 0, 0, im.width, h, im.frames); 
    while (1) {
        // combine the averages and differences
        Interleave::apply(win, 1, 2, 1);

        for (int t = 0; t < win.frames; t++) {            
            for (int y = 0; y < win.height; y+=2) {
                for (int x = 0; x < win.width; x++) {
                    float *a = win(x, y, t);
                    float *b = win(x, y+1, t);
                    for (int c = 0; c < win.channels; c++) {
                        float avg = a[c];
                        float diff = b[c];
                        a[c] = avg - diff/2;
                        b[c] = avg + diff/2;
                    }
                }
            }
        }
        // repeat
        h *= 2;
        if (h > im.height) break;
        win = Window(im, 0, 0, 0, im.width, h, im.frames);
    }

    // transform in x
    int w = 2*im.width/factor;
    win = Window(im, 0, 0, 0, w, im.height, im.frames); 
    while (1) {
        // combine the averages and differences
        Interleave::apply(win, 2, 1, 1);

        for (int t = 0; t < win.frames; t++) {            
            for (int y = 0; y < win.height; y++) {
                for (int x = 0; x < win.width; x+=2) {
                    float *a = win(x, y, t);
                    float *b = win(x+1, y, t);
                    for (int c = 0; c < win.channels; c++) {
                        float avg = a[c];
                        float diff = b[c];
                        a[c] = avg - diff/2;
                        b[c] = avg + diff/2;
                    }
                }
            }
        }
        // repeat
        w *= 2;
        if (w > im.width) break;
        win = Window(im, 0, 0, 0, w, im.height, im.frames);
    }


}




#define DAUB0 0.4829629131445341
#define DAUB1 0.83651630373780772
#define DAUB2 0.22414386804201339
#define DAUB3 -0.12940952255126034

void Daubechies::help() {
    printf("-daubechies performs the standard 2D daubechies 4 wavelet transform of an image. \n"
           "The image size must be a power of two.\n\n"
           "Usage: ImageStack -load in.jpg -daubechies -save out.jpg\n\n");
}

void Daubechies::parse(vector<string> args) {
    assert(args.size() == 0, "-daubechies takes no arguments");
    apply(stack(0));
}

void Daubechies::apply(Window im) {
    
    int i;
    for (i = 1; i < im.width; i <<= 1);
    assert(i == im.width, "Image width must be a power of two\n");
    for (i = 1; i < im.height; i <<= 1);
    assert(i == im.height, "Image height must be a power of two\n");

    // transform in x
    Window win(im, 0, 0, 0, im.width, im.height, im.frames); 
    while (1) {
        for (int t = 0; t < win.frames; t++) {            
            for (int y = 0; y < win.height; y++) {

                vector<float> saved1st(win.channels);
                vector<float> saved2nd(win.channels);
                for (int c = 0; c < win.channels; c++) {
                    saved1st[c] = win(0, y, t)[c];
                    saved2nd[c] = win(1, y, t)[c];
                }

                for (int x = 0; x < win.width-2; x+=2) {
                    float *a = win(x, y, t);
                    float *b = win(x+1, y, t);
                    float *c = win(x+2, y, t);
                    float *d = win(x+3, y, t);
                    for (int k = 0; k < win.channels; k++) {
                        float aVal = a[k];
                        float bVal = b[k];
                        float cVal = c[k];
                        float dVal = d[k];
                        a[k] = DAUB0 * aVal + DAUB1 * bVal + DAUB2 * cVal + DAUB3 * dVal;
                        b[k] = DAUB3 * aVal - DAUB2 * bVal + DAUB1 * cVal - DAUB0 * dVal;
                    }
                }
                // special case the last two elements using rotation
                float *a = win(win.width-2, y, t);
                float *b = win(win.width-1, y, t);
                float *c = &saved1st[0];
                float *d = &saved2nd[0];
                for (int k = 0; k < win.channels; k++) {
                    float aVal = a[k];
                    float bVal = b[k];
                    float cVal = c[k];
                    float dVal = d[k];
                    a[k] = DAUB0 * aVal + DAUB1 * bVal + DAUB2 * cVal + DAUB3 * dVal;
                    b[k] = DAUB3 * aVal - DAUB2 * bVal + DAUB1 * cVal - DAUB0 * dVal;
                }                
            }
        }
        // separate into averages and differences
        Deinterleave::apply(win, 2, 1, 1);
        
        if (win.width == 2) break;

        // repeat on the averages
        win = Window(win, 0, 0, 0, win.width/2, win.height, win.frames);
    }

    // transform in y
    win = Window(im, 0, 0, 0, im.width, im.height, im.frames); 
    while (1) {
        for (int t = 0; t < win.frames; t++) {            
            for (int x = 0; x < win.width; x++) {

                vector<float> saved1st(win.channels);
                vector<float> saved2nd(win.channels);
                for (int c = 0; c < win.channels; c++) {
                    saved1st[c] = win(x, 0, t)[c];
                    saved2nd[c] = win(x, 1, t)[c];
                }

                for (int y = 0; y < win.height-2; y+=2) {
                    float *a = win(x, y, t);
                    float *b = win(x, y+1, t);
                    float *c = win(x, y+2, t);
                    float *d = win(x, y+3, t);
                    for (int k = 0; k < win.channels; k++) {
                        float aVal = a[k];
                        float bVal = b[k];
                        float cVal = c[k];
                        float dVal = d[k];
                        a[k] = DAUB0 * aVal + DAUB1 * bVal + DAUB2 * cVal + DAUB3 * dVal;
                        b[k] = DAUB3 * aVal - DAUB2 * bVal + DAUB1 * cVal - DAUB0 * dVal;
                    }
                }
                // special case the last two elements using rotation
                float *a = win(x, win.height-2, t);
                float *b = win(x, win.height-1, t);
                float *c = &saved1st[0];
                float *d = &saved2nd[0];
                for (int k = 0; k < win.channels; k++) {
                    float aVal = a[k];
                    float bVal = b[k];
                    float cVal = c[k];
                    float dVal = d[k];
                    a[k] = DAUB0 * aVal + DAUB1 * bVal + DAUB2 * cVal + DAUB3 * dVal;
                    b[k] = DAUB3 * aVal - DAUB2 * bVal + DAUB1 * cVal - DAUB0 * dVal;
                }                
            }
        }
        // separate into averages and differences
        Deinterleave::apply(win, 1, 2, 1);
        
        if (win.height == 2) break;

        // repeat on the averages
        win = Window(win, 0, 0, 0, win.width, win.height/2, win.frames);
    }
}


void InverseDaubechies::help() {
    printf("-inversedaubechies performs the standard 2D daubechies 4 wavelet transform of an image. \n"
           "The image size must be a power of two.\n\n"
           "Usage: ImageStack -load in.jpg -inversedaubechies -save out.jpg\n\n");
}

void InverseDaubechies::parse(vector<string> args) {
    assert(args.size() == 0, "-inversedaubechies takes no arguments");
    apply(stack(0));
}

void InverseDaubechies::apply(Window im) {

    int i;
    for (i = 1; i < im.width; i <<= 1);
    assert(i == im.width, "Image width must be a power of two\n");
    for (i = 1; i < im.height; i <<= 1);
    assert(i == im.height, "Image height must be a power of two\n");

    
    // transform in x
    Window win(im, 0, 0, 0, 2, im.height, im.frames); 
    while (1) {
        // Collect averages and differences
        Interleave::apply(win, 2, 1, 1);

        for (int t = 0; t < win.frames; t++) {            
            for (int y = 0; y < win.height; y++) {
                vector<float> saved1st(win.channels);
                vector<float> saved2nd(win.channels);

                for (int c = 0; c < win.channels; c++) {
                    saved1st[c] = win(win.width-1, y, t)[c];
                    saved2nd[c] = win(win.width-2, y, t)[c];
                }

                for (int x = win.width-4; x >= 0; x-=2) {
                    float *a = win(x, y, t);
                    float *b = win(x+1, y, t);
                    float *c = win(x+2, y, t);
                    float *d = win(x+3, y, t);
                    for (int k = 0; k < win.channels; k++) {
                        float aVal = a[k];
                        float bVal = b[k];
                        float cVal = c[k];
                        float dVal = d[k];
                        c[k] = DAUB2 * aVal + DAUB1 * bVal + DAUB0 * cVal + DAUB3 * dVal;
                        d[k] = DAUB3 * aVal - DAUB0 * bVal + DAUB1 * cVal - DAUB2 * dVal;
                    }
                }
                // special case the first two elements using rotation
                float *a = &saved2nd[0];
                float *b = &saved1st[0];
                float *c = win(0, y, t);
                float *d = win(1, y, t);
                for (int k = 0; k < win.channels; k++) {
                    float aVal = a[k];
                    float bVal = b[k];
                    float cVal = c[k];
                    float dVal = d[k];
                    c[k] = DAUB2 * aVal + DAUB1 * bVal + DAUB0 * cVal + DAUB3 * dVal;
                    d[k] = DAUB3 * aVal - DAUB0 * bVal + DAUB1 * cVal - DAUB2 * dVal;
                }                
            }
        }
        
        if (win.width == im.width) break;

        // repeat on the averages
        win = Window(im, 0, 0, 0, win.width*2, win.height, win.frames);
    }

    // transform in y
    win = Window(im, 0, 0, 0, im.width, 2, im.frames); 
    while (1) {
        // Collect averages and differences
        Interleave::apply(win, 1, 2, 1);

        for (int t = 0; t < win.frames; t++) {            
            for (int x = 0; x < win.width; x++) {
                vector<float> saved1st(win.channels);
                vector<float> saved2nd(win.channels);

                for (int c = 0; c < win.channels; c++) {
                    saved1st[c] = win(x, win.height-1, t)[c];
                    saved2nd[c] = win(x, win.height-2, t)[c];
                }

                for (int y = win.height-4; y >= 0; y-=2) {
                    float *a = win(x, y, t);
                    float *b = win(x, y+1, t);
                    float *c = win(x, y+2, t);
                    float *d = win(x, y+3, t);
                    for (int k = 0; k < win.channels; k++) {
                        float aVal = a[k];
                        float bVal = b[k];
                        float cVal = c[k];
                        float dVal = d[k];
                        c[k] = DAUB2 * aVal + DAUB1 * bVal + DAUB0 * cVal + DAUB3 * dVal;
                        d[k] = DAUB3 * aVal - DAUB0 * bVal + DAUB1 * cVal - DAUB2 * dVal;
                    }
                }
                // special case the first two elements using rotation
                float *a = &saved2nd[0];
                float *b = &saved1st[0];
                float *c = win(x, 0, t);
                float *d = win(x, 1, t);
                for (int k = 0; k < win.channels; k++) {
                    float aVal = a[k];
                    float bVal = b[k];
                    float cVal = c[k];
                    float dVal = d[k];
                    c[k] = DAUB2 * aVal + DAUB1 * bVal + DAUB0 * cVal + DAUB3 * dVal;
                    d[k] = DAUB3 * aVal - DAUB0 * bVal + DAUB1 * cVal - DAUB2 * dVal;
                }                
            }
        }
        
        if (win.height == im.height) break;

        // repeat on the averages
        win = Window(im, 0, 0, 0, win.width, win.height*2, win.frames);
    }
}



#include "footer.h"
