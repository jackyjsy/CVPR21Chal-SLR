#include "main.h"
#include "File.h"
#include "header.h"

/* PPM file format:

1 A "magic number" for identifying the file type. A ppm image's magic number is the two characters "P6".

2 Whitespace (blanks, TABs, CRs, LFs).

3 A width, formatted as ASCII characters in decimal.

4 Whitespace.

5 A height, again in ASCII decimal.

6 Whitespace.

7 The maximum color value (Maxval), again in ASCII decimal. Must be less than 65536 and more than zero.

8 Newline or other single whitespace character.

9 A raster of Height rows, in order from top to bottom. Each row consists of Width pixels, in order from left to right. Each pixel is a triplet of red, green, and blue samples, in that order. Each sample is represented in pure binary by either 1 or 2 bytes. If the Maxval is less than 256, it is 1 byte. Otherwise, it is 2 bytes. The most significant byte is first.

A row of an image is horizontal. A column is vertical. The pixels in the image are square and contiguous.

10 In the raster, the sample values are "nonlinear." They are proportional to the intensity of the ITU-R Recommendation BT.709 red, green, and blue in the pixel, adjusted by the BT.709 gamma transfer function. (That transfer function specifies a gamma number of 2.2 and has a linear section for small intensities). A value of Maxval for all three samples represents CIE D65 white and the most intense color in the color universe of which the image is part (the color universe is all the colors in all images to which this image might be compared).

ITU-R Recommendation BT.709 is a renaming of the former CCIR Recommendation 709. When CCIR was absorbed into its parent organization, the ITU, ca. 2000, the standard was renamed. This document once referred to the standard as CIE Rec. 709, but it isn't clear now that CIE ever sponsored such a standard.

Note that another popular color space is the newer sRGB. A common variation on PPM is to subsitute this color space for the one specified.

11 Note that a common variation on the PPM format is to have the sample values be "linear," i.e. as specified above except without the gamma adjustment. pnmgamma takes such a PPM variant as input and produces a true PPM as output.

12 Characters from a "#" to the next end-of-line, before the maxval line, are comments and are ignored.
*/

namespace FilePPM {

    void help() {
        printf(".ppm files, of either 8 or 16 bit depth. When saving, an optional second\n"
               "argument, which defaults to 8, specifies the bit depth. ppm files always\n"
               "have three channels and one frame.\n");
    }

    Image load(string filename) {
        FILE *f = fopen(filename.c_str(), "rb");
        assert(f, "Could not open file %s", filename.c_str());
        
        // get the magic number
        assert(fgetc(f) == 'P' && fgetc(f) == '6', 
               "File does not start with ppm magic number: 'P6'");

        int width, height, maxval;
        assert(fscanf(f, " %d %d %d", &width, &height, &maxval) == 3, "Could not read image dimensions from ppm");

        // remove the next whitespace char
        fgetc(f);
        
        Image im(width, height, 1, 3);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < 3; c++) {
                    if (maxval > 255) {
                        im(x, y)[c] = (float)(((fgetc(f) & 255) << 8) + (fgetc(f) & 255)) / maxval;
                    } else {
                        im(x, y)[c] = (float)(fgetc(f)) / maxval;
                    }                    
                }
            }
        }

        fclose(f);

        return im;
    }

    void save(Window im, string filename, int depth) {
        FILE *f = fopen(filename.c_str(), "wb");
        assert(f, "Could not open file %s\n", filename.c_str());
        assert(depth == 16 || depth == 8, "bit depth must be 8 or 16\n");
        assert(im.frames == 1, "can only save single frame ppms\n");
        assert(im.channels == 3, "can only save three channel ppms\n");

        int maxval = (1 << depth) - 1;

        fprintf(f, "P6\n");
        fprintf(f, "%d %d\n%d\n", im.width, im.height, maxval);
        
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                for (int c = 0; c < 3; c++) {
                    float val = im(x, y)[c];
                    val = clamp(val, 0.0f, 1.0f);
                    val *= maxval;
                    if (maxval < 256) {
                        unsigned char v = (unsigned char)val;
                        fputc(v, f);
                    } else {
                        unsigned short v = (unsigned short)val;
                        fputc(v >> 8, f);
                        fputc(v & 255, f);
                    }
                }
            }
        }

        fclose(f);
    }
}
#include "footer.h"
