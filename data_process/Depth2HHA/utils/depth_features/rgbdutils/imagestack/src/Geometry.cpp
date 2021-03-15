#include "main.h"
#include "Geometry.h"
#include "Stack.h"
#include "Arithmetic.h"
#include "header.h"

void Upsample::help() {
    pprintf("-upsample multiplies the width, height, and frames of the current"
            " image by the given integer arguments. It uses nearest neighbor"
            " interpolation. For a slower, high-quality resampling method, use"
            " -resample instead.\n\n"
            "-upsample x y is interpreted as -upsample x y 1\n"
            "-upsample x is interpreted as -upsample x x 1\n"
            "-upsample is interpreted as -upsample 2 2 1\n\n"
            "Usage: ImageStack -load a.tga -upsample 3 2 -save b.tga\n\n");
}

void Upsample::parse(vector<string> args) {
    int boxWidth = 2, boxHeight = 2, boxFrames = 1;
    assert(args.size() <= 3, "-upsample takes three or fewer arguments\n");    
    if (args.size() == 3) {
        boxWidth = readInt(args[0]);
        boxHeight = readInt(args[1]);
        boxFrames = readInt(args[2]);
    } else if (args.size() == 2) {
        boxWidth = readInt(args[0]);
        boxHeight = readInt(args[1]);
    } else if (args.size() == 1) {
        boxWidth = boxHeight = readInt(args[0]);
    }

    Image im = apply(stack(0), boxWidth, boxHeight, boxFrames);
    pop();
    push(im);
}

Image Upsample::apply(Window im, int boxWidth, int boxHeight, int boxFrames) {

    int newWidth = im.width * boxWidth;
    int newHeight = im.height * boxHeight;
    int newFrames = im.frames * boxFrames;

    Image out(newWidth, newHeight, newFrames, im.channels);

    // huzzah for septuple nested for loops
    // the loop is ordered to maintain spatial coherence in the output
    float *outPtr = out(0, 0, 0);
    for (int t = 0; t < im.frames; t++) for (int dt = 0; dt < boxFrames; dt++) {
        for (int y = 0; y < im.height; y++) for (int dy = 0; dy < boxHeight; dy++) {
            float *imPtr = im(0, y, t);
            for (int x = 0; x < im.width; x++) {
                for (int dx = 0; dx < boxWidth; dx++) {
                    for (int c = 0; c < im.channels; c++) {
                        *outPtr++ = imPtr[c];
                    }
                }
                imPtr += im.channels;
            }
        }
    }

    return out;

}

void Downsample::help() {
    pprintf("-downsample divides the width, height, and frames of the current image"
            " by the given integer arguments. It averages rectangles to get the new"
            " values.\n\n"
            "-downsample x y is interpreted as -downsample x y 1\n"
            "-downsample x is interpreted as -downsample x x 1\n"
            "-downsample is interpreted as -downsample 2 2 1\n\n"
            "Usage: ImageStack -load a.tga -downsample 3 2 -save b.tga\n\n");
}

void Downsample::parse(vector<string> args) {
    int boxWidth = 2, boxHeight = 2, boxFrames = 1;
    assert(args.size() <= 3, "-downsample takes three or fewer arguments\n");
    if (args.size() == 3) {
        boxWidth = readInt(args[0]);
        boxHeight = readInt(args[1]);
        boxFrames = readInt(args[2]);
    } else if (args.size() == 2) {
        boxWidth = readInt(args[0]);
        boxHeight = readInt(args[1]);
    } else if (args.size() == 1) {
        boxWidth = boxHeight = readInt(args[0]);
    }

    Image im = apply(stack(0), boxWidth, boxHeight, boxFrames);
    pop();
    push(im);
}

Image Downsample::apply(Window im, int boxWidth, int boxHeight, int boxFrames) {

    if (!((im.width % boxWidth == 0) && (im.height % boxHeight == 0) && (im.frames % boxFrames == 0)))
        printf("Warning: Image dimensions are not a multiple of the downsample size. Ignoring some pixels.\n"); 

    int newWidth = im.width / boxWidth;
    int newHeight = im.height / boxHeight;
    int newFrames = im.frames / boxFrames;

    Image out(newWidth, newHeight, newFrames, im.channels);
    
    // this is all arranged for maximum spatial cache coherence for input
    for (int t = 0; t < newFrames; t++) for (int dt = 0; dt < boxFrames; dt++) {
        for (int y = 0; y < newHeight; y++) for (int dy = 0; dy < boxHeight; dy++) {
            float *imPtr = im(0, y*boxHeight+dy, t*boxFrames+dt);
            float *outPtr = out(0, y, t);
            for (int x = 0; x < newWidth; x++) {
                for (int dx = 0; dx < boxWidth; dx++) {
                    for (int c = 0; c < im.channels; c++) {                    
                        outPtr[c] += *imPtr++;
                    }
                }
                outPtr += out.channels;
            }
        }
    }

    Scale::apply(out, 1.0f/(boxWidth*boxHeight*boxFrames));

    return out;
}

void Resample::help() {
    printf("-resample resamples the input using a 3-lobed Lanczos filter. When"
           " given three arguments, it produces a new volume of the given width,"
           " height, and frames. When given two arguments, it produces a new volume"           
           " of the given width and height, with the same number of frames.\n\n"
           "Usage: ImageStack -loadframes f*.tga -resample 20 50 50 -saveframes f%%03d.tga\n\n");
}

void Resample::parse(vector<string> args) {

    if (args.size() == 2) {
        Image im = apply(stack(0), readInt(args[0]), readInt(args[1]));
        pop();
        push(im);
    } else if (args.size() == 3) {
        Image im = apply(stack(0), readInt(args[0]), readInt(args[1]), readInt(args[2]));
        pop();
        push(im);
    } else {
        panic("-resample takes two or three arguments\n");
    }
        
}

Image Resample::apply(Window im, int width, int height) {
    if (height != im.height && width != im.width) {
        Image tmp = resampleY(im, height);
        return resampleX(tmp, width); 
    } else if (width != im.width) {
        return resampleX(im, width);
    } else if (height != im.height) {
        return resampleY(im, height);
    }
    return im;
}

Image Resample::apply(Window im, int width, int height, int frames) {
    if (frames != im.frames) {
        Image tmp = resampleT(im, frames);
        return apply(tmp, width, height);
    } else {
        return apply(im, width, height);
    }
}

Image Resample::resampleX(Window im, int width) {
    float filterWidth = max(1.0f, (float)im.width / width);
    int filterBoxWidth = ((int)(filterWidth * 6 + 2) >> 1) << 1;

    Image out(width, im.height, im.frames, im.channels);

    for (int t = 0; t < out.frames; t++) {
        for (int y = 0; y < out.height; y++) {
            for (int x = 0; x < out.width; x++) {
                float oldX = ((float)x + 0.5) / width * im.width - 0.5;
                int oldXi = (int)floorf(oldX);
                int minX = max(0, oldXi - filterBoxWidth/2 + 1);
                int maxX = min(oldXi + filterBoxWidth/2, im.width-1);

                float totalWeight = 0;
                // iterate over the filter box
                for (int dx = minX; dx <= maxX; dx++) {
                    float weight = lanczos_3((dx - oldX)/filterWidth);
                    totalWeight += weight;
                    for (int c = 0; c < im.channels; c++) {
                        out(x, y, t)[c] += weight * im(dx, y, t)[c];
                    }
                }

                if (totalWeight > 0) {
                    for (int c = 0; c < im.channels; c++) out(x, y, t)[c] /= totalWeight;
                }
            }
        }
    }

    return out;  
}

Image Resample::resampleY(Window im, int height) {
    float filterHeight = max(1.0f, (float)im.height / height);
    int filterBoxHeight = ((int)(filterHeight * 6 + 2) >> 1) << 1;

    Image out(im.width, height, im.frames, im.channels);

    for (int t = 0; t < out.frames; t++) {
        for (int y = 0; y < out.height; y++) {
            float oldY = ((float)y + 0.5) / height * im.height - 0.5;
            int oldYi = (int)floorf(oldY);
            int minY = max(0, oldYi - filterBoxHeight/2 + 1);
            int maxY = min(oldYi + filterBoxHeight/2, im.height-1);

            for (int x = 0; x < out.width; x++) {
                float totalWeight = 0;
                // iterate over the filter box
                for (int dy = minY; dy <= maxY; dy++) {
                    float weight = lanczos_3((dy - oldY)/filterHeight);
                    totalWeight += weight;
                    for (int c = 0; c < im.channels; c++) {
                        out(x, y, t)[c] += weight * im(x, dy, t)[c];
                    }
                }

                if (totalWeight > 0) {
                    for (int c = 0; c < im.channels; c++) out(x, y, t)[c] /= totalWeight;
                }
            }
        }
    }

    return out;  
}

Image Resample::resampleT(Window im, int frames) {
    float filterFrames = max(1.0f, (float)im.frames / frames);
    int filterBoxFrames = ((int)(filterFrames * 6 + 2) >> 1) << 1;

    Image out(im.width, im.height, frames, im.channels);

    for (int t = 0; t < out.frames; t++) {
        float oldT = ((float)t + 0.5) / frames * im.frames - 0.5;
        int oldTi = (int)floorf(oldT);
        int minT = max(0, oldTi - filterBoxFrames/2 + 1);
        int maxT = min(oldTi + filterBoxFrames/2, im.frames-1);

        for (int y = 0; y < out.height; y++) {
            for (int x = 0; x < out.width; x++) {
                float totalWeight = 0;
                // iterate over the filter box
                for (int dt = minT; dt <= maxT; dt++) {
                    float weight = lanczos_3((dt - oldT)/filterFrames);
                    totalWeight += weight;
                    for (int c = 0; c < im.channels; c++) {
                        out(x, y, t)[c] += weight * im(x, y, dt)[c];
                    }
                }

                if (totalWeight > 0) {
                    for (int c = 0; c < im.channels; c++) out(x, y, t)[c] /= totalWeight;
                }
            }
        }
    }

    return out;  
}



void Interleave::help() {
    pprintf("-interleave divides the image into n equally sized volumes and interleaves"
            " them. When given two arguments it operates on columns and rows. When"
            " given three arguments, it operates on columns, rows, and frames.\n\n"
            "Usage: ImageStack -load deck.exr -interleave 2 -save shuffled.exr\n\n");
}

void Interleave::parse(vector<string> args) {
    if (args.size() == 2) {
        apply(stack(0), readInt(args[0]), readInt(args[1]));
    } else if (args.size() == 3) {
        apply(stack(0), readInt(args[0]), readInt(args[1]), readInt(args[2]));
    } else {
        panic("-interleave takes one, two, or three arguments\n");
    }
}

void Interleave::apply(Window im, int rx, int ry, int rt) {
    assert(rt >= 1 && rx >= 1 && ry >= 1, "arguments to interleave must be strictly positive integers\n");
    
    // interleave in t
    if (rt != 1) {
        Image tmp(1, 1, im.frames, im.channels);
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                // copy out this chunk
                for (int t = 0; t < im.frames; t++) {
                    for (int c = 0; c < im.channels; c++) {
                        tmp(0, 0, t)[c] = im(x, y, t)[c];
                    }
                }
                // paste this chunk back in in bizarro order
                int oldT = 0;
                for (int t = 0; t < im.frames; t++) {
                    for (int c = 0; c < im.channels; c++) {
                        im(x, y, oldT)[c] = tmp(0, 0, t)[c];
                    }
                    oldT += rt;
                    if (oldT >= im.frames) oldT = (oldT % rt) + 1;
                }
            }
        }
    }

    // interleave in x
    if (rx != 1) {
        Image tmp(im.width, 1, 1, im.channels);
        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {
                // copy out this chunk
                for (int x = 0; x < im.width; x++) {
                    for (int c = 0; c < im.channels; c++) {
                        tmp(x, 0, 0)[c] = im(x, y, t)[c];
                    }
                }

                // paste this chunk back in in bizarro order
                int oldX = 0;
                for (int x = 0; x < im.width; x++) {
                    for (int c = 0; c < im.channels; c++) {
                        im(oldX, y, t)[c] = tmp(x, 0, 0)[c];
                    }
                    oldX += rx;
                    if (oldX >= im.width) oldX = (oldX % rx) + 1;
                }
            }
        }
    }

    // interleave in y
    if (ry != 1) {
        Image tmp(1, im.height, 1, im.channels);
        for (int t = 0; t < im.frames; t++) {
            for (int x = 0; x < im.width; x++) {
                // copy out this chunk
                for (int y = 0; y < im.height; y++) {
                    for (int c = 0; c < im.channels; c++) {
                        tmp(0, y, 0)[c] = im(x, y, t)[c];
                    }
                }

                // paste this chunk back in in bizarro order
                int oldY = 0;
                for (int y = 0; y < im.height; y++) {
                    for (int c = 0; c < im.channels; c++) {
                        im(x, oldY, t)[c] = tmp(0, y, 0)[c];
                    }
                    oldY += ry;
                    if (oldY >= im.height) oldY = (oldY % ry) + 1;
                }
            }
        }
    }
}

void Deinterleave::help() {
    pprintf("-deinterleave collects every nth frame, column, and/or row of the image"
            " and tiles the resulting collections. When given two arguments it"
            " operates on columns and rows. When given three arguments, it operates"
            " on all columns, rows, and frames.\n\n"
            "Usage: ImageStack -load lf.exr -deinterleave 16 16 -save lftranspose.exr\n\n");
}

void Deinterleave::parse(vector<string> args) {
    if (args.size() == 2) {
        apply(stack(0), readInt(args[0]), readInt(args[1]));
    } else if (args.size() == 3) {
        apply(stack(0), readInt(args[0]), readInt(args[1]), readInt(args[2]));
    } else {
        panic("-deinterleave takes two or three arguments\n");
    }
}

void Deinterleave::apply(Window im, int rx, int ry, int rt) {
    assert(rt >= 1 && rx >= 1 && ry >= 1, "arguments to deinterleave must be strictly positive integers\n");
    
    // interleave in t
    if (rt != 1) {
        Image tmp(1, 1, im.frames, im.channels);
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                // copy out this chunk
                for (int t = 0; t < im.frames; t++) {
                    for (int c = 0; c < im.channels; c++) {
                        tmp(0, 0, t)[c] = im(x, y, t)[c];
                    }
                }
                // paste this chunk back in in bizarro order
                int oldT = 0;
                for (int t = 0; t < im.frames; t++) {
                    for (int c = 0; c < im.channels; c++) {
                        im(x, y, t)[c] = tmp(0, 0, oldT)[c];
                    }
                    oldT += rt;
                    if (oldT >= im.frames) oldT = (oldT % rt) + 1;
                }
            }
        }
    }

    // interleave in x
    if (rx != 1) {
        Image tmp(im.width, 1, 1, im.channels);
        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {
                // copy out this chunk
                for (int x = 0; x < im.width; x++) {
                    for (int c = 0; c < im.channels; c++) {
                        tmp(x, 0, 0)[c] = im(x, y, t)[c];
                    }
                }

                // paste this chunk back in in bizarro order
                int oldX = 0;
                for (int x = 0; x < im.width; x++) {
                    for (int c = 0; c < im.channels; c++) {
                        im(x, y, t)[c] = tmp(oldX, 0, 0)[c];
                    }
                    oldX += rx;
                    if (oldX >= im.width) oldX = (oldX % rx) + 1;
                }
            }
        }
    }

    // interleave in y
    if (ry != 1) {
        Image tmp(1, im.height, 1, im.channels);
        for (int t = 0; t < im.frames; t++) {
            for (int x = 0; x < im.width; x++) {
                // copy out this chunk
                for (int y = 0; y < im.height; y++) {
                    for (int c = 0; c < im.channels; c++) {
                        tmp(0, y, 0)[c] = im(x, y, t)[c];
                    }
                }

                // paste this chunk back in in bizarro order
                int oldY = 0;
                for (int y = 0; y < im.height; y++) {
                    for (int c = 0; c < im.channels; c++) {
                        im(x, y, t)[c] = tmp(0, oldY, 0)[c];
                    }
                    oldY += ry;
                    if (oldY >= im.height) oldY = (oldY % ry) + 1;
                }
            }
        }
    }
}


void Rotate::help() {
    printf("\n-rotate takes a number of degrees, and rotates every frame of the current image\n"
           "clockwise by that angle. The rotation preserves the image size, filling empty\n"
           " areas with zeros, and throwing away data which will not fit in the bounds.\n\n"
           "Usage: ImageStack -load a.tga -rotate 45 -save b.tga\n\n");
}


void Rotate::parse(vector<string> args) {
    assert(args.size() == 1, "-rotate takes one argument\n");
    Image im = apply(stack(0), readFloat(args[0]));
    pop();
    push(im);
}
    

Image Rotate::apply(Window im, float degrees) {    

    // figure out the rotation matrix
    float radians = degrees * M_PI / 180;
    float cosine = cosf(radians);
    float sine = sinf(radians);
    float m00 = cosine;
    float m01 = sine;
    float m10 = -sine;
    float m11 = cosine;

    // locate the origin
    float xorigin = (im.width-1) * 0.5;
    float yorigin = (im.height-1) * 0.5;

    Image out(im.width, im.height, im.frames, im.channels);

    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                // figure out the sample location
                float fx = m00 * (x - xorigin) + m01 * (y - yorigin) + xorigin;
                float fy = m10 * (x - xorigin) + m11 * (y - yorigin) + yorigin;
                // don't sample outside the image
                if (fx < 0 || fx > im.width || fy < 0 || fy > im.height) {
                    for (int i = 0; i < im.channels; i++) out(x, y, t)[i] = 0;
                } else {
                    im.sample2D(fx, fy, t, out(x, y, t));
                }
            }
        }
    }

    return out;

}


void AffineWarp::help() {
    printf("\n-affinewarp takes a 2x3 matrix in row major order, and performs that affine warp\n"
           "on the image.\n\n"
           "Usage: ImageStack -load a.jpg -affinewarp 0.9 0.1 0 0.1 0.9 0 -save out.jpg\n\n");
}

void AffineWarp::parse(vector<string> args) {
    assert(args.size() == 6, "-affinewarp takes six arguments\n");
    vector<double> matrix(6);
    for (int i = 0; i < 6; i++) matrix[i] = readFloat(args[i]);
    Image im = apply(stack(0), matrix);
    pop();
    push(im);
}

Image AffineWarp::apply(Window im, vector<double> matrix) {

    Image out(im.width, im.height, im.frames, im.channels);

    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                // figure out the sample location
                float fx = matrix[0] * x + matrix[1] * y + matrix[2] * im.width;
                float fy = matrix[3] * x + matrix[4] * y + matrix[5] * im.height;
                // don't sample outside the image
                if (fx < 0 || fx > im.width || fy < 0 || fy > im.height) {
                    for (int i = 0; i < im.channels; i++) out(x, y, t)[i] = 0;
                } else {
                    im.sample2D(fx, fy, t, out(x, y, t));
                }
            }
        }
    }

    return out;
}

void Crop::help() {
    pprintf("-crop takes either zero, two, four, or six arguments. The first half"
            " of the arguments are either minimum t, minimum x and y, or all three"
            " in the order x, y, t. The second half of the arguments are"
            " correspondingly either number of frames, width and height, or all"
            " three in the order width, height, frames. You may crop outside the"
            " bounds of the original image. Values there are assumed to be black. If"
            " no argument are given, ImageStack guesses how much to crop by trimming"
            " rows and columns that are all the same color as the top left"
            " pixel.\n\n"
            "Usage: ImageStack -loadframes f*.tga -crop 10 1 -save frame10.tga\n"
            "       ImageStack -load a.tga -crop 100 100 200 200 -save cropped.tga\n"
            "       ImageStack -loadframes f*.tga -crop 100 100 10 200 200 1\n"
            "                  -save frame10cropped.tga\n\n");
}

void Crop::parse(vector<string> args) {
     
    Image im;

    if (args.size() == 0) {
        im = apply(stack(0));
    } else if (args.size() == 2) {
        im = apply(stack(0), 
                   0, 0, readInt(args[0]), 
                   stack(0).width, stack(0).height, readInt(args[1]));
    } else if (args.size() == 4) {
        im = apply(stack(0), 
                   readInt(args[0]), readInt(args[1]),
                   readInt(args[2]), readInt(args[3]));
    } else if (args.size() == 6) {
        im = apply(stack(0), 
                   readInt(args[0]), readInt(args[1]), readInt(args[2]),
                   readInt(args[3]), readInt(args[4]), readInt(args[5]));
    } else {
        panic("-crop takes two, four, or six arguments.\n");
    }

    pop();
    push(im);
}

Image Crop::apply(Window im) {
    int minX, maxX, minY, maxY, minT, maxT;

    // calculate minX
    for (minX = 0; minX < im.width; minX++) {
        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {
                for (int c = 0; c < im.channels; c++) {
                    if (im(minX, y, t)[c] != im(0, 0, 0)[c]) goto minXdone;
                }
            }
        }
    }
  minXdone:

    // calculate maxX
    for (maxX = im.width-1; maxX >= 0; maxX--) {
        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {
                for (int c = 0; c < im.channels; c++) {
                    if (im(maxX, y, t)[c] != im(0, 0, 0)[c]) goto maxXdone;
                }
            }
        }
    }
  maxXdone:

    // calculate minY
    for (minY = 0; minY < im.height; minY++) {
        for (int t = 0; t < im.frames; t++) {
            for (int x = 0; x < im.width; x++) {
                for (int c = 0; c < im.channels; c++) {
                    if (im(x, minY, t)[c] != im(0, 0, 0)[c]) goto minYdone;
                }
            }
        }
    }
  minYdone:

    // calculate maxY
    for (maxY = im.height-1; maxY >= 0; maxY--) {
        for (int t = 0; t < im.frames; t++) {
            for (int x = 0; x < im.width; x++) {
                for (int c = 0; c < im.channels; c++) {
                    if (im(x, maxY, t)[c] != im(0, 0, 0)[c]) goto maxYdone;
                }
            }
        }
    }
  maxYdone:

    // calculate minT
    for (minT = 0; minT < im.frames; minT++) {
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                for (int c = 0; c < im.channels; c++) {
                    if (im(x, y, minT)[c] != im(0, 0, 0)[c]) goto minTdone;
                }
            }
        }
    }
  minTdone:

    // calculate maxT
    for (maxT = im.frames-1; maxT >= 0; maxT--) {
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                for (int c = 0; c < im.channels; c++) {
                    if (im(x, y, maxT)[c] != im(0, 0, 0)[c]) goto maxTdone;
                }
            }
        }
    }
  maxTdone:

    int width = maxX - minX + 1;
    int height = maxY - minY + 1;
    int frames = maxT - minT + 1;
    
    assert(width >= 0 && height >= 0 && frames >= 0, "Can't auto crop a blank image\n");

    return apply(im, minX, minY, minT, width, height, frames);

}

Image Crop::apply(Window im, int minX, int minY, int width, int height) {    
    return apply(im,
                 minX, minY, 0, 
                 width, height, im.frames);
}


Image Crop::apply(Window im, int minX, int minY, int minT,
                  int width, int height, int frames) {         
    Image out(width, height, frames, im.channels);

    for (int t = max(0, -minT); t < min(frames, im.frames - minT); t++) {
        for (int y = max(0, -minY); y < min(height, im.height - minY); y++) {
            for (int x = max(0, -minX); x < min(width, im.width - minX); x++) {
                for (int c = 0; c < im.channels; c++) {
                    out(x, y, t)[c] = im(x + minX, y + minY, t + minT)[c];
                }
            }
        }
    }

    return out;
}

void Flip::help() {
    printf("-flip takes 'x', 'y', or 't' as the argument and flips the current image along\n"
           "that dimension.\n\n"
           "Usage: ImageStack -load a.tga -flip x -save reversed.tga\n\n");
}

void Flip::parse(vector<string> args) {
    assert(args.size() == 1, "-flip takes exactly one argument\n");
    char dimension = readChar(args[0]);
    apply(stack(0), dimension);
}

void Flip::apply(Window im, char dimension) {    
    if (dimension == 't') {
        for (int t = 0; t < im.frames/2; t++) {
            for (int y = 0; y < im.height; y++) {
                for (int x = 0; x < im.width; x++) {                    
                    float *ptr1 = im(x, y, t);
                    float *ptr2 = im(x, y, im.frames - t - 1);
                    for (int c = 0; c < im.channels; c++) {
                        float tmp = ptr1[c];
                        ptr1[c] = ptr2[c];
                        ptr2[c] = tmp;                        
                    }
                }
            }
        }
    } else if (dimension == 'y') { 
        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height/2; y++) {
                for (int x = 0; x < im.width; x++) {                    
                    float *ptr1 = im(x, y, t);
                    float *ptr2 = im(x, im.height - 1 - y, t);
                    for (int c = 0; c < im.channels; c++) {
                        float tmp = ptr1[c];
                        ptr1[c] = ptr2[c];
                        ptr2[c] = tmp;                        
                    }
                }
            }
        }
    } else if (dimension == 'x') { 
        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {
                for (int x = 0; x < im.width/2; x++) {                    
                    float *ptr1 = im(x, y, t);
                    float *ptr2 = im(im.width - 1 - x, y, t);
                    for (int c = 0; c < im.channels; c++) {
                        float tmp = ptr1[c];
                        ptr1[c] = ptr2[c];
                        ptr2[c] = tmp;                        
                    }
                }
            }
        }
    } else {
        panic("-flip only understands dimensions 'x', 'y', and 't'\n");
    }
}


void Adjoin::help() {
    printf("\n-adjoin takes 'x', 'y', 't', or 'c' as the argument, and joins the top two\n"
           "images along that dimension. The images must match in the other dimensions.\n\n"
           "Usage: ImageStack -load a.tga -load b.tga -adjoin x -save ab.tga\n\n");
}

void Adjoin::parse(vector<string> args) {
    assert(args.size() == 1, "-adjoin takes exactly one argument\n");
    char dimension = readChar(args[0]);
    Image im = apply(stack(1), stack(0), dimension);
    pop();
    pop();
    push(im);
}


Image Adjoin::apply(Window a, Window b, char dimension) {
    int newFrames = a.frames, newWidth = a.width, newHeight = a.height, newChannels = a.channels;
    int tOff = 0, xOff = 0, yOff = 0, cOff = 0;

    if (dimension == 't') {
        assert(a.width    == b.width &&
               a.height   == b.height &&
               a.channels == b.channels,
               "Cannot adjoin images that don't match in other dimensions\n");
        tOff = newFrames;
        newFrames += b.frames;
    } else if (dimension == 'y') {
        assert(a.width    == b.width &&
               a.frames   == b.frames &&
               a.channels == b.channels,
               "Cannot adjoin images that don't match in other dimensions\n");
        yOff = newHeight;
        newHeight += b.height;
    } else if (dimension == 'c') {
        assert(a.frames == b.frames &&
               a.height == b.height &&
               a.width  == b.width,
               "Cannot adjoin images that don't match in other dimensions\n");
        cOff = newChannels;
        newChannels += b.channels;
    } else if (dimension == 'x') {
        assert(a.frames == b.frames &&
               a.height == b.height &&
               a.channels  == b.channels,
               "Cannot adjoin images that don't match in other dimensions\n");
        xOff = newWidth;
        newWidth += b.width;
    } else {
        panic("-adjoin only understands dimensions 'x', 'y', and 't'\n");
    }

    Image out(newWidth, newHeight, newFrames, newChannels);
    // paste in the first image
    for (int t = 0; t < a.frames; t++) {
        for (int y = 0; y < a.height; y++) {
            for (int x = 0; x < a.width; x++) {
                for (int c = 0; c < a.channels; c++) {
                    out(x, y, t)[c] = a(x, y, t)[c];
                }
            }
        }
    }
    // paste in the second image
    for (int t = 0; t < b.frames; t++) {
        for (int y = 0; y < b.height; y++) {
            for (int x = 0; x < b.width; x++) {
                for (int c = 0; c < b.channels; c++) {
                    out(x + xOff, y + yOff, t + tOff)[c + cOff] = b(x, y, t)[c];
                }
            }
        }
    }

    return out;
}

void Transpose::help() {
    printf("-transpose takes two dimension of the form 'x', 'y', or 't' and transposes\n"
           "the current image over those dimensions. If given no arguments, it defaults\n"
           "to x and y.\n\n"
           "Usage: ImageStack -load a.tga -transpose x y -flip x -save rotated.tga\n\n");
}

void Transpose::parse(vector<string> args) {
    assert(args.size() == 0 || args.size() == 2, "-transpose takes either zero or two arguments\n");
    if (args.size() == 0) {
        Image im = apply(stack(0), 'x', 'y');
        pop();
        push(im);
    } else {
        char arg1 = readChar(args[0]);
        char arg2 = readChar(args[1]);
        Image im = apply(stack(0), arg1, arg2);
        pop();
        push(im);
    }
}


Image Transpose::apply(Window im, char arg1, char arg2) {
    
    char dim1 = min(arg1, arg2);
    char dim2 = max(arg1, arg2);
    
    if (dim1 == 'c' && dim2 == 'y') {
        Image out(im.width, im.channels, im.frames, im.height);
        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {
                for (int x = 0; x < im.width; x++) {
                    for (int c = 0; c < im.channels; c++) {
                        out(x, c, t)[y] = im(x, y, t)[c];
                    }
                }
            }
        }
        return out;
    } else if (dim1 == 'c' && dim2 == 't') {
        Image out(im.width, im.height, im.channels, im.frames);
        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {
                for (int x = 0; x < im.width; x++) {
                    for (int c = 0; c < im.channels; c++) {
                        out(x, y, c)[t] = im(x, y, t)[c];
                    }
                }
            }
        }
        return out;
    } else if (dim1 == 'c' && dim2 == 'x') {
        Image out(im.channels, im.height, im.frames, im.width);
        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {
                for (int x = 0; x < im.width; x++) {
                    for (int c = 0; c < im.channels; c++) {
                        out(c, y, t)[x] = im(x, y, t)[c];
                    }
                }
            }
        }
        return out;
    } else if (dim1 == 'x' && dim2 == 'y') {

        Image out(im.height, im.width, im.frames, im.channels);
        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {
                for (int x = 0; x < im.width; x++) {
                    for (int c = 0; c < im.channels; c++) {
                        out(y, x, t)[c] = im(x, y, t)[c];
                    }
                }
            }
        }
        return out;

    } else if (dim1 == 't' && dim2 == 'x') {
        Image out(im.frames, im.height, im.width, im.channels);
        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {
                for (int x = 0; x < im.width; x++) {
                    for (int c = 0; c < im.channels; c++) {
                        out(t, y, x)[c] = im(x, y, t)[c];
                    }
                }
            }
        }
        return out;
    } else if (dim1 == 't' && dim2 == 'y') {
        Image out(im.width, im.frames, im.height, im.channels);
        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {
                for (int x = 0; x < im.width; x++) {
                    for (int c = 0; c < im.channels; c++) {
                        out(x, t, y)[c] = im(x, y, t)[c];
                    }
                }
            }
        }
        return out;
    } else {
        panic("-transpose only understands dimensions 'c', 'x', 'y', and 't'\n");
    }

    // keep compiler happy
    return Image();

}

void Translate::help() {
    printf("\n-translate moves the image data, leaving black borders. It takes two or three\n"
           "arguments. Two arguments are interpreted as a shift in x and a shift in y. Three\n"
           "arguments indicates a shift in t, x, and y. Negative values shift to the top left\n"
           "and positive ones to the bottom right. The unit is pixels.\n\n"
           "Usage: ImageStack -load in.jpg -translate -10 -10 -translate 20 20\n"
           "                  -translate -10 -10 -save in_border.jpg\n\n");
}

void Translate::parse(vector<string> args) {
    if (args.size() == 2) {
        Image im = apply(stack(0), readInt(args[0]), readInt(args[1]), 0);
        pop();
        push(im);
    } else if (args.size() == 3) {
        Image im = apply(stack(0), readInt(args[0]), readInt(args[1]), readInt(args[2]));
        pop();
        push(im);
    } else {
        panic("-translate requires two or three arguments\n");
    }
}

Image Translate::apply(Window im, int dx, int dy, int dt) {
    return Crop::apply(im, -dx, -dy, -dt, im.width, im.height, im.frames);
}

void Paste::help() {
    printf("-paste places some of the second image in the stack inside the top image, at\n"
           "the specified location. -paste accepts two or three, six, or nine arguments.\n"
           "When given two or three arguments, it interprets these as x and y, or x, y,\n"
           "and t, and pastes the whole of the second image onto that location in the first\n"
           "image. If six or nine arguments are given, the latter four or six arguments\n"
           "specify what portion of the second image is copied. The middle two or three\n"
           "arguments specify the top left, and the last two or three arguments specify\n"
           "the size of the region to paste.\n\n"
           "The format is thus: -paste [desination origin] [source origin] [size]\n\n"
           "Usage: ImageStack -load a.jpg -push 820 820 1 3 -paste 10 10 -save border.jpg\n\n");
}
    
void Paste::parse(vector<string> args) {
    int xdst = 0, ydst = 0, tdst = 0;
    int xsrc = 0, ysrc = 0, tsrc = 0;
    int width = stack(1).width;
    int height = stack(1).height;
    int frames = stack(1).frames;

    if (args.size() == 2) {
        xdst = readInt(args[0]);
        ydst = readInt(args[1]);
    } else if (args.size() == 3) {
        xdst = readInt(args[0]);
        ydst = readInt(args[1]);
        tdst = readInt(args[2]);
    } else if (args.size() == 6) {
        xdst = readInt(args[0]);
        ydst = readInt(args[1]);
        xsrc = readInt(args[2]);
        ysrc = readInt(args[3]);
        width = readInt(args[4]);
        height = readInt(args[5]);
    } else if (args.size() == 9) {
        xdst = readInt(args[0]);
        ydst = readInt(args[1]);
        tdst = readInt(args[2]);
        xsrc = readInt(args[3]);
        ysrc = readInt(args[4]);
        tsrc = readInt(args[5]);
        width  = readInt(args[6]);
        height = readInt(args[7]);            
        frames = readInt(args[8]);
    } else {
        panic("-paste requires two, three, six, or nine arguments\n");
    }

    apply(stack(0), stack(1), 
          xdst, ydst, tdst, 
          xsrc, ysrc, tsrc, 
          width, height, frames);
    pull(1);
    pop();

}


void Paste::apply(Window into, Window from, 
                  int xdst, int ydst, 
                  int xsrc, int ysrc, 
                  int width, int height) {
    apply(into, from,
          xdst, ydst, 0,
          xsrc, ysrc, 0,
          width, height, from.frames);
}

void Paste::apply(Window into, Window from, 
                  int xdst, int ydst, int tdst) {
    apply(into, from,
          xdst, ydst, tdst,
          0, 0, 0,
          from.width, from.height, from.frames);
}

void Paste::apply(Window into, Window from, 
                  int xdst, int ydst, int tdst, 
                  int xsrc, int ysrc, int tsrc, 
                  int width, int height, int frames) {
    assert(into.channels == from.channels,
           "Images must have the same number of channels\n");
    assert(tdst >= 0 &&
           ydst >= 0 &&
           xdst >= 0 &&
           tdst + frames <= into.frames &&
           ydst + height <= into.height &&
           xdst + width  <= into.width, 
           "Cannot paste outside the target image\n");
    assert(tsrc >= 0 &&
           ysrc >= 0 &&
           xsrc >= 0 &&
           tsrc + frames <= from.frames &&
           ysrc + height <= from.height &&
           xsrc + width  <= from.width,
           "Cannot paste from outside the source image\n");
    for (int t = 0; t < frames; t++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < into.channels; c++) {
                    into(x + xdst, y + ydst, t + tdst)[c] =
                        from(x + xsrc, y + ysrc, t + tsrc)[c];
                }
            }
        }
    }
}

void Tile::help() {
    printf("\n-tile repeats the image along each dimension. It interprets two arguments as\n"
           "repetitions in x and y. Three arguments are interpreted as repetitions in x,\n"
           "y, and t.\n\n"
           "Usage: ImageStack -load a.tga -tile 2 2 -save b.tga\n\n");
}

void Tile::parse(vector<string> args) {
    int tRepeat = 1, xRepeat = 1, yRepeat = 1;
    if (args.size() == 2) {
        xRepeat = readInt(args[0]);
        yRepeat = readInt(args[1]);
    } else if (args.size() == 3) {
        xRepeat = readInt(args[0]);
        yRepeat = readInt(args[1]);
        tRepeat = readInt(args[2]);
    } else {
        panic("-tile takes two or three arguments\n");
    }
    Image im = apply(stack(0), xRepeat, yRepeat, tRepeat);
    pop();
    push(im);
}

Image Tile::apply(Window im, int xRepeat, int yRepeat, int tRepeat) {

    Image out(im.width * xRepeat, im.height * yRepeat, im.frames * tRepeat, im.channels);
    
    for (int t = 0; t < im.frames * tRepeat; t++) {
        int imT = t % im.frames;
        for (int y = 0; y < im.height * yRepeat; y++) {
            int imY = y % im.height;
            for (int x = 0; x < im.width * xRepeat; x++) {
                int imX = x % im.width;
                for (int c = 0; c < im.channels; c++) {
                    out(x, y, t)[c] = im(imX, imY, imT)[c];
                }
            }
        }
    }

    return out;
}


void Subsample::help() {
    printf("\n-subsample subsamples the current image. Given two integer arguments, a and b,\n"
           "it selects one out of every a frames starting from frame b. Given four arguments,\n"
           "a, b, c, d, it selects one pixel out of every axb sized box, starting from pixel\n"
           "(c, d). Given six arguments, a, b, c, d, e, f, it selects one pixel from every\n"
           "axbxc volume, in the order width, height, frames starting at pixel (d, e, f).\n\n"
           "Usage: ImageStack -load in.jpg -subsample 2 2 0 0 -save smaller.jpg\n\n");
}

void Subsample::parse(vector<string> args) {
    if (args.size() == 2) {
        Image im = apply(stack(0), readInt(args[0]), readInt(args[1]));
        pop(); push(im);
    } else if (args.size() == 4) {
        Image im = apply(stack(0), readInt(args[0]), readInt(args[1]), 
                         readInt(args[2]), readInt(args[3]));
        pop(); push(im);        
    } else if (args.size() == 6) {
        Image im = apply(stack(0), readInt(args[0]), readInt(args[1]), readInt(args[2]),
                         readInt(args[3]), readInt(args[4]), readInt(args[5]));
        pop(); push(im);
    } else {
        panic("-subsample needs two, four, or six arguments\n");
    }
}

Image Subsample::apply(Window im, int boxFrames, int offsetT) {
    return apply(im, 1, 1, boxFrames, 0, 0, offsetT);
}

Image Subsample::apply(Window im, int boxWidth, int boxHeight, 
                       int offsetX, int offsetY) {
    return apply(im, boxWidth, boxHeight, 1, offsetX, offsetY, 0);
}

Image Subsample::apply(Window im, int boxWidth, int boxHeight, int boxFrames,
                      int offsetX, int offsetY, int offsetT) {

    int newFrames = 0, newWidth = 0, newHeight = 0;
    for (int t = offsetT; t < im.frames; t += boxFrames) newFrames++;
    for (int x = offsetX; x < im.width;  x += boxWidth)  newWidth++;
    for (int y = offsetY; y < im.height; y += boxHeight) newHeight++;
    
    Image out(newWidth, newHeight, newFrames, im.channels);

    int outT = 0;
    for (int t = offsetT; t < im.frames; t += boxFrames) {
        int outY = 0;
        for (int y = offsetY; y < im.height; y += boxHeight) {
            int outX = 0;
            for (int x = offsetX; x < im.width; x += boxWidth) {
                for (int c = 0; c < im.channels; c++) {
                    out(outX, outY, outT)[c] = im(x, y, t)[c];
                }
                outX++;
            }
            outY++;
        }
        outT++;
    }

    return out;
}

void TileFrames::help() {
    printf("\n-tileframes takes a volume and lays down groups of frames in a grid, dividing\n"
           "the number of frames by the product of the arguments. It takes two arguments,\n"
           "the number of old frames across each new frame, and the number of frames down.\n"
           "each new frame. The first batch of frames will appear as the first row of the.\n"
           "first frame of the new volume.\n\n"
           "Usage: ImageStack -loadframes frame*.tif -tileframes 5 5 -saveframes sheet%%d.tif\n\n");
}

void TileFrames::parse(vector<string> args) {
    assert(args.size() == 2, "-tileframes takes two arguments\n");

    Image im = apply(stack(0), readInt(args[0]), readInt(args[1]));
    pop();
    push(im);
}

Image TileFrames::apply(Window im, int xTiles, int yTiles) {

    int newWidth = im.width * xTiles;
    int newHeight = im.height * yTiles;
    int newFrames = (int)(ceil((float)im.frames / (xTiles * yTiles)));

    Image out(newWidth, newHeight, newFrames, im.channels);

    for (int t = 0; t < newFrames; t++) {
        int outY = 0;
        for (int yt = 0; yt < yTiles; yt++) {            
            for (int y = 0; y < im.height; y++) {
                int outX = 0;
                for (int xt = 0; xt < xTiles; xt++) {
                    int imT = (t * yTiles + yt) * xTiles + xt;
                    if (imT >= im.frames) break;
                    for (int x = 0; x < im.width; x++) {
                        for (int c = 0; c < im.channels; c++) {
                            out(outX, outY, t)[c] = im(x, y, imT)[c];
                                
                        }
                        outX++;
                    }
                }
                outY++;
            }
        }
    }        

    return out;
}

void FrameTiles::help() {
    printf("\n-frametiles takes a volume where each frame is a grid and breaks each frame\n"
           "into multiple frames, one per grid element. The two arguments specify the grid\n"
           "size. This operation is the inverse of tileframes.\n\n"
           "Usage: ImageStack -loadframes sheet*.tif -frametiles 5 5 -saveframes frame%%d.tif\n\n");
}

void FrameTiles::parse(vector<string> args) {
    assert(args.size() == 2, "-frametiles takes two arguments\n");

    Image im = apply(stack(0), readInt(args[0]), readInt(args[1]));
    pop();
    push(im);
}

Image FrameTiles::apply(Window im, int xTiles, int yTiles) {

    assert(im.width % xTiles == 0 &&
           im.height % yTiles == 0,
           "The image is not divisible by the given number of tiles\n");

    int newWidth = im.width / xTiles;
    int newHeight = im.height / yTiles;
    int newFrames = im.frames * xTiles * yTiles;

    Image out(newWidth, newHeight, newFrames, im.channels);

    for (int t = 0; t < im.frames; t++) {
        int imY = 0;
        for (int yt = 0; yt < yTiles; yt++) {            
            for (int y = 0; y < newHeight; y++) {
                int imX = 0;
                for (int xt = 0; xt < xTiles; xt++) {
                    int outT = (t * yTiles + yt) * xTiles + xt;
                    for (int x = 0; x < newWidth; x++) {
                        for (int c = 0; c < im.channels; c++) {
                            out(x, y, outT)[c] = im(imX, imY, t)[c];
                        }
                        imX++;
                    }
                }
                imY++;
            }
        }
    }        

    return out;
}


void Warp::help() {
    printf("\n-warp treats the top image of the stack as indices (within [0, 1]) into the\n"
           "second image, and samples the second image accordingly. It takes no arguments.\n"
           "The number of channels in the top image is the dimensionality of the warp, and\n"
           "should be three or less.\n\n"
           "Usage: ImageStack -load in.jpg -push -evalchannels \"X+Y\" \"Y\" -warp -save out.jpg\n\n");
}

void Warp::parse(vector<string> args) {
    assert(args.size() == 0, "warp takes no arguments\n");
    Image im = apply(stack(0), stack(1));
    pop();
    pop();
    push(im);
}

Image Warp::apply(Window coords, Window source) {    

    Image out(coords.width, coords.height, coords.frames, source.channels);

    if (coords.channels == 3) {
        for (int t = 0; t < coords.frames; t++) {
            for (int y = 0; y < coords.height; y++) {
                for (int x = 0; x < coords.width; x++) {
                    float *srcCoords = coords(x, y, t);
                    source.sample3D(srcCoords[0]*source.width, 
                                     srcCoords[1]*source.height,
                                     srcCoords[2]*source.frames, 
                                     out(x, y, t));
                }
            }
        }
    } else if (coords.channels == 2) {
        for (int t = 0; t < coords.frames; t++) {
            for (int y = 0; y < coords.height; y++) {
                for (int x = 0; x < coords.width; x++) {
                    float *srcCoords = coords(x, y, t);
                    source.sample2D(srcCoords[0]*source.width, 
                                     srcCoords[1]*source.height, 
                                     t, out(x, y, t));
                }
            }
        }
    } else {
        panic("index image must have two or three channels\n");
    }
    return out;
}    



void Reshape::help() {
    printf("\n-reshape changes the way the memory of the current image is indexed. The four\n"
           "integer arguments specify a new width, height, frames, and channels.\n\n"
           "Usage: ImageStack -load movie.tmp -reshape width height*frames 1 channels\n"
           "                  -save filmstrip.tmp\n");
}

void Reshape::parse(vector<string> args) {
    assert(args.size() == 4, "-reshape takes four arguments\n");
    Image im = apply(stack(0), 
                     readInt(args[0]), readInt(args[1]),
                     readInt(args[2]), readInt(args[3]));
    pop();
    push(im);
    
}

Image Reshape::apply(Window im, int x, int y, int t, int c) {
    panic("Reshape can only be applied to images, not more general windows\n");
    return Image();
}

Image Reshape::apply(Image im, int x, int y, int t, int c) {
    assert(t * x * y * c == im.frames * im.width * im.height * im.channels,
           "New shape uses a different amount of memory that the old shape.\n");
    Image out(im);
    out.frames = t;
    out.width = x;
    out.height = y;
    out.channels = c;
    out.xstride = c;
    out.ystride = c*x;
    out.tstride = c*x*y;
    return out;
}



#include "footer.h"
