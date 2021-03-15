#include "main.h"
#include "Color.h"
#include "Arithmetic.h"
#include "header.h"

void ColorMatrix::help() {
    printf("\n-colormatrix treats each pixel as a vector over its channels and multiplies\n"
           "the vector by the given matrix. The matrix size and shape is deduced from the\n"
           "number of arguments. The matrix is specified in column major order.\n\n"
           "Converting rgb to grayscale:\n"
           "  ImageStack -load color.tga -colormatrix 1 1 1 -scale 0.33333 -save gray.tga\n\n"
           "Making orange noise:\n"
           "  ImageStack -push 100 100 1 1 -noise -colormatrix 1 0.3 0 -save noise.tga\n\n"
           "Making noise that varies between orange and blue:\n"
           "  ImageStack -push 100 100 1 2 -noise -colormatrix 1 0.3 0 0 0 1 -save noise.tga\n\n");
}

void ColorMatrix::parse(vector<string> args) {
    assert(args.size() > 0, "-colormatrix requires arguments\n");

    vector<float> matrix(args.size());
    for (size_t i = 0; i < args.size(); i++) {
        matrix[i] = readFloat(args[i]);
    }

    Image im = apply(stack(0), matrix);
    pop();
    push(im);
}

Image ColorMatrix::apply(Window im, vector<float> matrix) {
    assert(matrix.size() % im.channels == 0, 
           "-colormatrix requires a number of arguments that is a multiple of the number of\n"
           "channels of the current image\n");

    int inChannels = im.channels;
    int outChannels = (int)matrix.size() / inChannels;

    Image out(im.width, im.height, im.frames, outChannels);

    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                for (int c = 0; c < im.channels; c++) {
                    for (int i = 0; i < out.channels; i++) {
                        out(x, y, t)[i] += im(x, y, t)[c] * matrix[c * outChannels + i];
                    }
                }
            }
        }
    }

    return out;
}




void ColorConvert::help() {
    printf("\n-colorconvert converts from one colorspace to another. It is called with two\n"
           "arguments representing these colorspaces.\n\n"
           "Allowable colorspaces are rgb, yuv, hsv, xyz, lab and y (luminance alone). grayscale,\n"
           "gray, and luminance are synonyms for y, and hsb and hsl are synonyms for hsv.\n\n"
           "Usage: ImageStack -load a.tga -colorconvert rgb hsv -scale 0.1 1 1\n"
           "                  -colorconvert hsv rgb -save out.tga\n\n");
}

void ColorConvert::parse(vector<string> args) {
    assert(args.size() == 2, "-colorconvert requires two arguments\n");
    Image im = apply(stack(0), args[0], args[1]);
    pop();
    push(im);
}

Image ColorConvert::apply(Window im, string from, string to) {
    // check for the trivial case
    assert(from != to, "color conversion from %s to %s is pointless\n", from.c_str(), to.c_str());

    // unsupported destination color spaces
    if (to == "yuyv" ||
        to == "uyvy") {
        panic("Unsupported destination color space: %s\n", to.c_str());
    }

    // direct conversions that don't have to go via rgb
    if (from == "yuyv" && to == "yuv") {        
        return yuyv2yuv(im);
    } else if (from == "uyvy" && to == "yuv") {
        return uyvy2yuv(im);
    } else if (from == "xyz" && to == "lab") {
        return xyz2lab(im);
    } else if (from == "lab" && to == "xyz") {
        return lab2xyz(im);
    } else if (from != "rgb" && to != "rgb") {
        // conversions that go through rgb
        Image halfway = apply(im, from, "rgb");
        return apply(halfway, "rgb", to);
    } else if (from == "rgb") { // from rgb
        if (to == "hsv" || to == "hsl" || to == "hsb") {
            return rgb2hsv(im);
        } else if (to == "yuv") {
            return rgb2yuv(im);
        } else if (to == "xyz") {
            return rgb2xyz(im);
        } else if (to == "y" || to == "gray" || 
                   to == "grayscale" || to == "luminance") {
            return rgb2y(im);
        } else if (to == "lab"){
            return rgb2lab(im);
        } else {
            panic("Unknown color space %s\n", to.c_str());
        }
    } else { //(to == "rgb") 
        if (from == "hsv" || from == "hsl" || from == "hsb") {
            return hsv2rgb(im);
        } else if (from == "yuv") {
            return yuv2rgb(im);
        } else if (from == "xyz") {
            return xyz2rgb(im);
        } else if (from == "y" || from == "gray" || 
                   from == "grayscale" || from == "luminance") {
            return y2rgb(im);
        } else if (from == "lab") {
            return lab2rgb(im);
        } else if (from == "uyvy") {
            return uyvy2rgb(im);
        } else if (from == "yuyv") {
            return yuyv2rgb(im);
        } else {
            panic("Unknown color space %s\n", from.c_str());
        }
    }

    // keep the compiler happy
    return Image();

}

//conversions to and from lab inspired by CImg (http://cimg.sourceforge.net/) 
Image ColorConvert::xyz2lab(Window im){
    #define labf(x)  ((x)>=0.008856?(powf(x,1/3.0)):(7.787*(x)+16.0/116.0))
        
    assert(im.channels == 3, "Image does not have 3 channels\n");
    
    Image out(im.width, im.height, im.frames, im.channels);
    
    //left in this form to allow for changes/fine-tuning
    float Xn = 1.0f/(0.412453 + 0.357580 + 0.180423);
    float Yn = 1.0f/(0.212671 + 0.715160 + 0.072169);
    float Zn = 1.0f/(0.019334 + 0.119193 + 0.950227);
    
    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                float L, a, b;
                float X = im(x, y, t)[0];
                float Y = im(x, y, t)[1];
                float Z = im(x, y, t)[2];
                L = 1.16f * labf(Y*Yn) - 0.16f;
                a = 5.0f * (labf(X*Xn) - labf(Y*Yn));
                b = 2.0f * (labf(Y*Yn) - labf(Z*Zn));
                out(x, y, t)[0] = L;
                out(x, y, t)[1] = a;
                out(x, y, t)[2] = b;
            }
        }
    }
    return out;
    
}

Image ColorConvert::lab2xyz(Window im){
    assert(im.channels == 3, "Image does not have 3 channels\n");
    Image out(im.width, im.height, im.frames, im.channels);
    
    float s = 6.0/29;
    
    float Xn = 0.412453 + 0.357580 + 0.180423;
    float Yn = 0.212671 + 0.715160 + 0.072169;
    float Zn = 0.019334 + 0.119193 + 0.950227;
    
    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                float X, Y, Z, fy, fx, fz;
                float L = im(x, y, t)[0];
                float a = im(x, y, t)[1];
                float b = im(x, y, t)[2];
                fy = (L + 0.16f)/1.16f;
                fx = fy + a/5.0f;
                fz = fy - b/2.0f;
                
                if (fy > s) {
                    Y = Yn*(fy * fy * fy);
                } else {
                    Y = (fy - 16.0/116)*3*s*s*Yn;
                }
                if (fx > s) {
                    X = Xn*(fx * fx * fx);
                } else {
                    X = (fx - 16.0/116)*3*s*s*Xn;
                }
                if (fz > s) {
                    Z = Zn*(fz * fz * fz);
                } else {
                    Z = (fz - 16.0/116)*3*s*s*Zn;
                }
                
                out(x, y, t)[0] = X;
                out(x, y, t)[1] = Y;
                out(x, y, t)[2] = Z;
            }
        }
    }
    
    return out;
}

Image ColorConvert::rgb2lab(Window im){
    assert(im.channels == 3, "Image does not have 3 channels\n");
    
    return xyz2lab(rgb2xyz(im));
}

Image ColorConvert::lab2rgb(Window im){
    assert(im.channels == 3, "Image does not have 3 channels\n");
    return xyz2rgb(lab2xyz(im));
}


Image ColorConvert::rgb2hsv(Window im) {
    assert(im.channels == 3, "Image does not have 3 channels\n");

    float mult = 1.0f / 6;

    Image out(im.width, im.height, im.frames, im.channels);

    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                float minV, maxV, delta;
                float h, s, v;
                float r = im(x, y, t)[0];
                float g = im(x, y, t)[1];
                float b = im(x, y, t)[2];

                minV = min(r, g, b);
                maxV = max(r, g, b);
                v = maxV;   
                
                delta = maxV - minV;
                
                if (delta != 0) {
                    s = delta / maxV; 
                    if (r == maxV)      h = 0 + (g - b) / delta; // between yellow & magenta
                    else if (g == maxV) h = 2 + (b - r) / delta; // between cyan & yellow
                    else                h = 4 + (r - g) / delta; // between magenta & cyan
                    h *= mult;
                    if (h < 0) h++;
                } else {
                    // r = g = b = 0 so s = 0, h is undefined
                    s = 0;
                    h = 0;
                }
                
                out(x, y, t)[0] = h;
                out(x, y, t)[1] = s;
                out(x, y, t)[2] = v;
            }
        }
    }

    return out;
}

Image ColorConvert::hsv2rgb(Window im) {
    assert(im.channels == 3, "Image does not have 3 channels\n");

    Image out(im.width, im.height, im.frames, im.channels);

    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                float h = im(x, y, t)[0];
                float s = im(x, y, t)[1];
                float v = im(x, y, t)[2];
                
                if (s == 0) {
                    // achromatic (grey)
                    im(x, y, t)[0] = im(x, y, t)[1] = im(x, y, t)[2] = v;
                } else {
                    
                    h *= 6;        // sector 0 to 5
                    int i = (int)h;
                    if (i == 6) i = 5;
                    float f = h - i;
                    float p = v * (1 - s);
                    float q = v * (1 - s * f);
                    float u = v * (1 - s * (1 - f));
                    
                    switch (i) {
                    case 0:
                        out(x, y, t)[0] = v;
                        out(x, y, t)[1] = u;
                        out(x, y, t)[2] = p;
                        break;
                    case 1:
                        out(x, y, t)[0] = q;
                        out(x, y, t)[1] = v;
                        out(x, y, t)[2] = p;
                        break;
                    case 2:
                        out(x, y, t)[0] = p;
                        out(x, y, t)[1] = v;
                        out(x, y, t)[2] = u;
                        break;
                    case 3:
                        out(x, y, t)[0] = p;
                        out(x, y, t)[1] = q;
                        out(x, y, t)[2] = v;
                        break;
                    case 4:
                        out(x, y, t)[0] = u;
                        out(x, y, t)[1] = p;
                        out(x, y, t)[2] = v;
                        break;
                    default:  // case 5:
                        out(x, y, t)[0] = v;
                        out(x, y, t)[1] = p;
                        out(x, y, t)[2] = q;
                        break;
                    }
                }
            }
        }
    }

    return out;
}

Image ColorConvert::rgb2y(Window im) {
    assert(im.channels == 3, "Image does not have 3 channels\n");

    Image out(im.width, im.height, im.frames, 1);

    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                out(x, y, t)[0] = (im(x, y, t)[0] * 0.299f + 
                                   im(x, y, t)[1] * 0.587f + 
                                   im(x, y, t)[2] * 0.114f);
            }
        }
    }

    return out;

}

Image ColorConvert::y2rgb(Window im) {        
    assert(im.channels == 1, "Image does not have one channel\n");

    Image out(im.width, im.height, im.frames, 3);

    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                out(x, y, t)[2] = out(x, y, t)[1] = out(x, y, t)[0] = im(x, y, t)[0];
            }
        }
    }

    return out;
}

Image ColorConvert::rgb2yuv(Window im) {
    assert(im.channels == 3, "Image does not have 3 channels\n");
    Image out(im.width, im.height, im.frames, 3);

    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                float r = im(x, y, t)[0];
                float g = im(x, y, t)[1];
                float b = im(x, y, t)[2];
                out(x, y, t)[0] = r *  0.299f + g *  0.587f + b *  0.114f;
                out(x, y, t)[1] = r * -0.169f + g * -0.332f + b *  0.500f + 0.5f;
                out(x, y, t)[2] = r *  0.500f + g * -0.419f + b * -0.0813f + 0.5f;
            }
        }
    }

    return out;

}

Image ColorConvert::yuv2rgb(Window im) {
    assert(im.channels == 3, "Image does not have 3 channels\n");

    Image out(im.width, im.height, im.frames, 3);

    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                float Y = im(x, y, t)[0];
                float U = im(x, y, t)[1] - 0.5f;
                float V = im(x, y, t)[2] - 0.5f;
                out(x, y, t)[0] = Y + 1.4075f * V;
                out(x, y, t)[1] = Y - 0.3455f * U - 0.7169f * V;
                out(x, y, t)[2] = Y + 1.7790f * U;
            }
        }
    }

    return out;
}

Image ColorConvert::rgb2xyz(Window im) {
    assert(im.channels == 3, "Image does not have 3 channels\n");

    float mat[9] = {0.412453, 0.212671, 0.019334,
                    0.357580, 0.715160, 0.119193,
                    0.180423, 0.072169, 0.950227};
    vector<float> matrix(mat, mat+9);
    
    return ColorMatrix::apply(im, matrix);

}

Image ColorConvert::xyz2rgb(Window im) {
    assert(im.channels == 3, "Image does not have 3 channels\n");

    float mat[9] = {3.240479, -0.969256, 0.055648,
                    -1.537150, 1.875992, -0.204043,
                    -0.498535, 0.041556, 1.057311};
    vector<float> matrix(mat, mat+9);

    return ColorMatrix::apply(im, matrix);

}

Image ColorConvert::uyvy2yuv(Window im) {
    assert(im.channels == 2, 
           "uyvy images should be stored as a two channel image where the second"
           " channel represents luminance (y), and the first channel alternates"
           " between u and v.\n");
    assert((im.width & 1) == 0,
           "uyvy images must have an even width\n");

    Image out(im.width, im.height, im.frames, 3);
    float *outPtr = out(0, 0, 0);  
    for (int t = 0; t < out.frames; t++) {
        for (int y = 0; y < out.height; y++) {
            float *imPtr = im(0, y, t);
            for (int x = 0; x < out.width; x+=2) {
                *outPtr++ = imPtr[1]; // Y 
                *outPtr++ = imPtr[0]; // U 
                *outPtr++ = imPtr[2]; // V
                *outPtr++ = imPtr[3]; // Y 
                *outPtr++ = imPtr[0]; // U 
                *outPtr++ = imPtr[2]; // V
                imPtr += 4;
            }
        }
    }
    
    return out;
}

Image ColorConvert::yuyv2yuv(Window im) {
    assert(im.channels == 2, 
           "yuyv images should be stored as a two channel image where the first"
           " channel represents luminance (y), and the second channel alternates"
           " between u and v.\n");
    assert((im.width & 1) == 0,
           "uyvy images must have an even width\n");
    
    Image out(im.width, im.height, im.frames, 3);
    float *outPtr = out(0, 0, 0);  
    for (int t = 0; t < out.frames; t++) {
        for (int y = 0; y < out.height; y++) {
            float *imPtr = im(0, y, t);
            for (int x = 0; x < out.width; x+=2) {
                *outPtr++ = imPtr[0]; // Y 
                *outPtr++ = imPtr[1]; // U 
                *outPtr++ = imPtr[3]; // V
                *outPtr++ = imPtr[2]; // Y 
                *outPtr++ = imPtr[1]; // U 
                *outPtr++ = imPtr[3]; // V
                imPtr += 4;
            }
        }
    }
    
    return out;
}

Image ColorConvert::uyvy2rgb(Window im) {
    return yuv2rgb(uyvy2yuv(im));
}

Image ColorConvert::yuyv2rgb(Window im) {
    return yuv2rgb(yuyv2yuv(im));    
}

//TODO: rgb2lab
//lab2rgb
//xyz2lab
//lab2xyz




void Demosaic::help() {
    printf("\n-demosaic demosaics a raw bayer mosaiced image camera. It should be a one\n"
           "channel image. The algorithm used is adaptive color plane interpolation (ACPI).\n"
           "Demosaic optionally takes two or three arguments. Two arguments specify an offset\n"
           "of the standard bayer pattern in x and y. The presence of a third argument\n"
           "indicates that auto-white-balancing should be performed.\n\n"
           "Usage: ImageStack -load photo.dng -demosaic -save out.png\n"
           "       ImageStack -load raw.yuv -demosaic 0 1 awb -save out.png\n");
}

void Demosaic::parse(vector<string> args) {
    bool awb = false;
    int xoff = 0, yoff = 0;
    if (args.size() == 0) {
        awb = false;
    } else if (args.size() == 2) {
        xoff = readInt(args[0]);
        yoff = readInt(args[1]);
    } else if (args.size() == 3) {
        xoff = readInt(args[0]);
        yoff = readInt(args[1]);
        awb = true;
    } else {
        panic("-demosaic takes zero, two, or three arguments");        
    }
    Image im = apply(stack(0), xoff, yoff, awb);
    pop();
    push(im);
}

Image Demosaic::apply(Window im, int xoff, int yoff, bool awb) {

    assert(im.channels == 1, "Mosaiced images should have a single channel\n");

    Image out(im.width, im.height, im.frames, 3);

    // This algorithm is adaptive color plane interpolation (ACPI)
    // by Runs Hamilton and Adams 
    // (The Adams is not me)

    // make sure the image is of even width and height
    if (im.width & 1 || im.height & 1) {
        im = Window(im, 0, 0, 0, (im.width >> 1) << 1, (im.width >> 1) << 1, im.frames);
    }

    if (awb) {

        // Step 1
        // auto white balance: make sure all channels have the same mean
        double sum[2][2] = {{0, 0}, {0, 0}};
        double maximum[2][2] = {{0, 0}, {0, 0}};
        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {
                for (int x = 0; x < im.width; x++) {
                    double val = im(x, y, t)[0];
                    maximum[x & 1][y & 1] = max(maximum[x & 1][y & 1], val);
                    sum[x & 1][y & 1] += val;
                }                
            }
        }
        
        double scale = sum[0][0]/maximum[0][0];
        double multiplier[2][2] = {{1.0/maximum[0][0], scale/sum[0][1]}, 
                                   {scale/sum[1][0],   scale/sum[1][1]}};
        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {
                for (int x = 0; x < im.width; x++) {
                    im(x, y, t)[0] = (float)(im(x, y, t)[0] * multiplier[x & 1][y & 1]);
                }
            }
        }
    }

    // Step 2
    // Interpolate green, blending in horizontal or vertical directions depending on 
    // gradient magnitudes. Add a correction factor based on the second derivative of
    // the color channel at that point.
    // Ie, calculate |dI/dx| + |d2I/dx2| and |dI/dy| + |d2I/dy2|, and interpolate
    // horizontally or vertically depending on which is smaller
    // if they're both the same, use both
    for (int t = 0; t < im.frames; t++) {
        for (int y = 2; y < im.height-2; y++) {
            for (int x = 2; x < im.width-2; x++) {
                if (((x + xoff) & 1) == ((y + yoff) & 1)) {
                    // GREEN IS KNOWN
                    // we already know green here, just use it
                    out(x, y, t)[1] = im(x, y, t)[0];
                } else {
                    // RED OR BLUE IS KNOWN
                    // gather neighbouring greens
                    float left1 = im(x-1, y, t)[0], right1 = im(x+1, y, t)[0];
                    float up1 = im(x, y-1, t)[0], down1 = im(x, y+1, t)[0];
                    // gather neighbouring reds or blues
                    float here = im(x, y, t)[0];
                    float left2 = im(x-2, y, t)[0], right2 = im(x+1, y, t)[0];
                    float up2 = im(x, y-2, t)[0], down2 = im(x, y+2, t)[0];
                    
                    // decide which way to interpolate 
                    // (divide laplacian by two because it's across twice the baseline)
                    // the correction terms have been removed because they look retarded
                    float interpHoriz = fabs(right1 - left1) + fabs(2*here - right2 - left2)/2;
                    float interpVert  = fabs(up1    - down1) + fabs(2*here - up2    - down2)/2;
                    if (interpHoriz < interpVert) { // horizontally
                        //float colAverage = (left2 + right2)/2;
                        //float correction = here - colAverage;
                        // only apply half the correction, because it's across twice the baseline
                        out(x, y, t)[1] = (left1 + right1)/2;// + correction/2; 
                    } else if (interpVert < interpHoriz) { // vertically
                        //float colAverage = (up2 + down2)/2;
                        //float correction = here - colAverage;
                        out(x, y, t)[1] = (up1 + down1)/2;// + correction/2;
                    } else { // both
                        float colAverage = (up2 + down2 + left2 + right2)/4;
                        float correction = here - colAverage;
                        out(x, y, t)[1] = (left1 + up1 + right1 + down1)/4 + correction/2;
                    }
                }
            }
        }
    }
    
    // Step 3
    // to predict blue (or red) on top of green, 
    // A) average the two neighbouring blue (or red) pixels
    // B) Calculate the error you would have made if you did the same thing to predict green
    // C) Correct blue (or red) by that error

    // Step 4
    // to predict blue on red or red on blue,
    // we have 4 neighbours, diagonally around us
    // use the same approach as step 2, but take diagonal derivatives and interpolate diagonally
   
    for (int t = 0; t < im.frames; t++) {
        for (int y = 2; y < im.height-2; y++) {
            for (int x = 2; x < im.width-2; x++) {
                if (((x + xoff) & 1) == ((y + yoff) & 1)) { 
                    // GREEN IS KNOWN (step 3)

                    // figure out which of red/blue is horizontally interpolated
                    // and which is vertically interpolated
                    int horizChannel, vertChannel;
                    if ((y + yoff) & 1) {
                        horizChannel = 2;
                        vertChannel = 0;
                    } else {
                        horizChannel = 0;
                        vertChannel = 2;
                    }

                    // do the horizontal interpolation

                    // compute an average for the color
                    float colLeft = im(x-1, y, t)[0], colRight = im(x+1, y, t)[0];
                    float colAverage = (colLeft + colRight)/2;
                    // compute the same average for green
                    float greenLeft = out(x-1, y, t)[1], greenRight = out(x+1, y, t)[1], greenHere = out(x, y, t)[1];
                    float greenAverage = (greenLeft + greenRight)/2;
                    // see how wrong the green average was
                    float correction = greenHere - greenAverage; 
                    // set the output to the average color plus the correction factor needed for green
                    out(x, y, t)[horizChannel] = colAverage + correction;
                    
                    // do the vertical interpolation
                    float colUp = im(x, y-1, t)[0], colDown = im(x, y+1, t)[0];
                    float greenUp = out(x, y-1, t)[1], greenDown = out(x, y+1, t)[1];
                    colAverage = (colUp + colDown)/2;
                    greenAverage = (greenUp + greenDown)/2;
                    correction = greenHere - greenAverage;
                    out(x, y, t)[vertChannel] = colAverage + correction;
                    
                } else {
                    // RED OR BLUE IS KNOWN (step 4)
                    
                    // figure out which channel is known exactly
                    int knownChannel, unknownChannel;
                    if ((y+yoff) & 1) {
                        knownChannel = 2;
                        unknownChannel = 0;
                    } else {
                        knownChannel = 0;
                        unknownChannel = 2;
                    }
                    
                    // set the known channel to the correct value
                    out(x, y, t)[knownChannel] = im(x, y, t)[0];
                    
                    // for the unknown channel, do diagonal interpolation
                    // u is up left, v is down right, s is up right, t is down left
                    // p is the channel to be predicted, g is green (already interpolated)
                    float up = im(x-1, y-1, t)[0], ug = out(x-1, y-1, t)[1];
                    float vp = im(x+1, y+1, t)[0], vg = out(x+1, y+1, t)[1];
                    float sp = im(x+1, y-1, t)[0], sg = out(x+1, y-1, t)[1];
                    float tp = im(x-1, y+1, t)[0], tg = out(x-1, t+1, t)[1];
                    float greenHere = out(x, y, t)[1];
                    
                    float interpUV = fabs(vp - up) + fabs(2*greenHere - vg - ug);
                    float interpST = fabs(sp - tp) + fabs(2*greenHere - sg - tg);
                    
                    if (interpUV < interpST) {
                        float greenAverage = (ug + vg)/2;
                        float correction = greenHere - greenAverage;
                        out(x, y, t)[unknownChannel] = (up + vp)/2 + correction;
                    } else if (interpST < interpUV) {
                        float greenAverage = (sg + tg)/2;
                        float correction = greenHere - greenAverage;
                        out(x, y, t)[unknownChannel] = (sp + tp)/2 + correction;
                    } else {
                        float greenAverage = (ug + vg + sg + tg)/4;
                        float correction = greenHere - greenAverage;
                        out(x, y, t)[unknownChannel] = (up + vp + sp + tp)/4 + correction;
                    }
                    
                }
            }
        }
    }


    // Step 5
    // zero the margins, which weren't interpolated, to avoid annoying checkerboard there
    // we could also do some more basic interpolation, but the margins don't really matter anyway
    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            if (y == 0 || y == 1 || y == im.height - 2 || y == im.height - 1) {
                for (int x = 0; x < im.width; x++) {
                    out(x, y, t)[0] = out(x, y, t)[1] = out(x, y, t)[2] = 0;
                }
            } else {
                for (int c = 0; c < 6; c++) {
                    out(0, y, t)[0] = out(0, y, t)[1] = out(0, y, t)[2] = 0;
                    out(1, y, t)[0] = out(1, y, t)[1] = out(1, y, t)[2] = 0;
                    out(im.width-2, y, t)[0] = out(im.width-2, y, t)[1] = out(im.width-2, y, t)[2] = 0;
                    out(im.width-1, y, t)[0] = out(im.width-1, y, t)[1] = out(im.width-1, y, t)[2] = 0;
                }
            }
        }
    }


    // Step 6
    // clamp the output between zero and one
    // ET 2008-08-28: This works quite poorly when dealing with HDR-composited mosaiced images.  Commenting out
    // Clamp::apply(out, 0, 1);


    return out;
}

#include "footer.h"
