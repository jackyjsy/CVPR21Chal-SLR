#include "main.h"
#include "Panorama.h"
#include "File.h"
#include "Stack.h"
#include "header.h"

void LoadPanorama::help() {
    printf("\n-loadpanorama takes a filename as its first argument. The file must be the\n"
           "homography text file output from autostitch. It loads and parses this file, \n"
           "and places each warped image in a separate frame. The remaining six arguments\n"
           "specify minimum and maximum theta, then phi, then the desired output resolution.\n\n"
            "Usage: ImageStack -loadpanorama pano.txt -0.1 0.1 -0.1 0.1 640 480 -display\n\n");
} 

void LoadPanorama::parse(vector<string> args) {
    assert(args.size() == 7, "-loadpanorama takes seven arguments\n");
    push(apply(args[0], 
               readFloat(args[1]), readFloat(args[2]),
               readFloat(args[3]), readFloat(args[4]),
               readInt(args[5]), readInt(args[6])));
}

Image LoadPanorama::apply(string filename, 
                          float minTheta, float maxTheta,
                          float minPhi, float maxPhi,
                          int width, int height) {
    FILE *pano = fopen(filename.c_str(), "rb");
    assert(pano, "Could not open file %s\n", filename.c_str());

    char fname[4096]; 
    char line[4096]; 
    // scan through once counting the number of images
    int frames = 0;
    while (fgets(fname, 4095, pano) != NULL) {
        // get the dimensions line
        assert(fgets(line, 4095, pano) != NULL, "unexpected EOF\n");
        // get the blank line
        assert(fgets(line, 4095, pano) != NULL, "unexpected EOF\n");
        // get the matrix
        for (int i = 0; i < 3; i++) {
            assert(fgets(line, 4095, pano) != NULL, "unexpected EOF\n");
        }
        // get the next blank line
        assert(fgets(line, 4095, pano) != NULL, "unexpected EOF\n");
        // get the next matrix
        for (int i = 0; i < 3; i++) {
            assert(fgets(line, 4095, pano) != NULL, "unexpected EOF\n");
        }
        // get the next blank line
        assert(fgets(line, 4095, pano) != NULL, "unexpected EOF\n");
        // get the focal distance
        assert(fgets(line, 4095, pano) != NULL, "unexpected EOF\n");
        // get the last blank line
        assert(fgets(line, 4095, pano) != NULL, "unexpected EOF\n");            
        frames++;
    } 

    // allocate the memory for the frames
    Image im(width, height, frames, 4);

    float Tmatrix[3][3];
    float Rmatrix[3][3];
    float focalDistance;
    float matrix[3][3];

    // read them in
    fseek(pano, 0, SEEK_SET);
    int t = 0;
    while (fgets(fname, 4095, pano) != NULL) {
        fname[strlen(fname)-2] = '\0'; // trim the newline and carraige return autostitch writes
        printf("%s\n", fname);
        // get the dimensions line
        assert(fgets(line, 4095, pano) != NULL, "unexpected EOF\n");
        // get the blank line
        assert(fgets(line, 4095, pano) != NULL, "unexpected EOF\n");

        // load the frame
        Image next = Load::apply(fname);
        // get the T matrix
         for (int i = 0; i < 3; i++) {
            assert(fgets(line, 4095, pano) != NULL, "unexpected EOF\n");
            sscanf(line, "%f %f %f", 
                    &Tmatrix[i][0], &Tmatrix[i][1], &Tmatrix[i][2]);
             
        }

        // get the next blank line
        assert(fgets(line, 4095, pano) != NULL, "unexpected EOF\n");

         // get the R matrix
        for (int i = 0; i < 3; i++) {
            assert(fgets(line, 4095, pano) != NULL, "unexpected EOF\n");

             sscanf(line, "%f %f %f", 
                    &Rmatrix[i][0], &Rmatrix[i][1], &Rmatrix[i][2]);

        }

        // get the next blank line
        assert(fgets(line, 4095, pano) != NULL, "unexpected EOF\n");

        // get the focal distance
        assert(fgets(line, 4095, pano) != NULL, "unexpected EOF\n");
        sscanf(line, "%f", &focalDistance);

        // get the last blank line
        assert(fgets(line, 4095, pano) != NULL, "unexpected EOF\n");            



        // calculate the matrix (T * K(f) * R)
        for (int i = 0; i < 3; i++) {
            Rmatrix[0][i] *= focalDistance;
            Rmatrix[1][i] *= focalDistance;
        }

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                matrix[i][j] = 0;
                for (int k = 0; k < 3; k++) {
                    matrix[i][j] += Tmatrix[i][k] * Rmatrix[k][j];
                }
            }
        }

        assert(next.channels == 3, "Input image does not have 3 channels");
 
        // add an alpha channel that tapers at the edges
        Image nextWithAlpha(next.width, next.height, 1, 4);
        for (int y = 0; y < next.height; y++) {
            for (int x = 0; x < next.width; x++) {
                for (int c = 0; c < next.channels; c++) {
                    nextWithAlpha(x, y)[c] = next(x, y)[c];
                }
                nextWithAlpha(x, y)[3] = (min(1.0f, 0.05f * min(x, next.width - 1 - x)) *
                                          min(1.0f, 0.05f * min(y, next.height - 1 - y)));
            }
        }

        float dTheta = (maxTheta - minTheta) / width;
        float dPhi = -(maxPhi - minPhi) / height;

        // warp the image into the output
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                float theta = (float)x * dTheta + minTheta;
                float phi = (float)y * dPhi + maxPhi;
                
                float X = -sin(phi); 
                float Y = sin(theta) * cos(phi);
                float Z = cos(theta) * cos(phi);
                
                float W = 1.0f / (matrix[2][0] * X + matrix[2][1] * Y + matrix[2][2] * Z);
                
                float srcY = (matrix[0][0] * X + matrix[0][1] * Y + matrix[0][2] * Z) * W;
                float srcX = (matrix[1][0] * X + matrix[1][1] * Y + matrix[1][2] * Z) * W;
                nextWithAlpha.sample2D(srcX, srcY, im(x, y, t));
            }
        }

    
        t++;
    }

    return im;
}


void PanoramaBackground::help() {
    printf("\n-panoramabackground takes aligned frames and computes what lies behind any\n"
           "objects that move across the frames.\n\n"
           "Usage: ImageStack -loadpanorama pano.txt -panoramabackground -display\n\n");
}

void PanoramaBackground::parse(vector<string> args) {
    assert(args.size() == 0, "-panoramabackground takes no arguments\n");
    Image im = apply(stack(0));
    pop();
    push(im);
}

Image PanoramaBackground::apply(Window im) {
    assert(im.frames > 1 && im.channels == 4, "-panoramabackground requires a 4 channel multiframe image\n");

    Image out(im.width, im.height, 1, im.channels);

    vector< vector<float> > samples(im.frames);
    for (int t = 0; t < im.frames; t++) {
        samples[t] = vector<float>(im.channels + 1);
    }

    for (int y = 0; y < im.height; y++) {
        for (int x = 0; x < im.width; x++) {

            // gather the samples
            for (int t = 0; t < im.frames; t++) {
                for (int c = 0; c < im.channels; c++) {
                    samples[t][c] = im(x, y, t)[c];
                }
                // set the initial weight to the alpha value
                samples[t][im.channels] = samples[t][im.channels - 1];

            }

            // iterate a weighted average
            float totalWeight = 0;

            for (int i = 0; ; i++) {
                // find the weighted average
                totalWeight = 0;
                for (int c = 0; c < im.channels; c++) {
                    out(x, y)[c] = 0;
                }
                for (int t = 0; t < im.frames; t++) {
                    float weight = samples[t][im.channels];
                    totalWeight += weight;
                    for (int c = 0; c < im.channels; c++) {
                        out(x, y)[c] += weight * samples[t][c];
                    }
                }

                if (totalWeight > 0) {
                    for (int c = 0; c < im.channels; c++) {
                        out(x, y)[c] /= totalWeight;
                    }
                } else break;

                if (i > 5) break;

                // recompute the weights as the distance from the mean times the alpha value
                for (int t = 0; t < im.frames; t++) {
                    float distance = 0;
                    for (int c = 0; c < im.channels; c++) {
                        float difference = samples[t][c] - out(x, y)[c];
                        distance += difference * difference;
                    }
                    samples[t][im.channels] = expf(-5*distance) * samples[t][im.channels - 1];
                }

            }
        }
    }

    return out;
}



#include "footer.h"
