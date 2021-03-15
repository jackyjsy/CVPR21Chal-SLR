#include "main.h"
#include "File.h"
#include "header.h"

namespace FileTGA {

    /*
    typedef struct {
        unsigned char  identsize;          // size of ID field that follows 18 byte header (0 usually)
        unsigned char  colormaptype;       // type of colour map 0=none, 1=has palette
        unsigned char  imagetype;          // type of image 0=none,1=indexed,2=rgb,3=grey,+8=rle packed
        char colormap[5];                  // crap to do with the color map
        short xstart;                      // image x origin
        short ystart;                      // image y origin
        short width;                       // image width in pixels
        short height;                      // image height in pixels
        unsigned char  bits;               // image bits per pixel 8,16,24,32
        unsigned char  descriptor;         // image descriptor bits (vh flip bits)
    } Header;
    */

    void help() {
        printf(".tga files. These can have 1, 3, or 4 channels, are run-length encoded, and\n"
               "are low dynamic range.\n");
    }

    Image load(string filename) {
        FILE *f = fopen(filename.c_str(), "rb");
        assert(f, "Could not open file %s\n", filename.c_str());

        unsigned char identsize, colormaptype, imagetype, bits, descriptor;
        int xstart, ystart, width, height;

        identsize = fgetc(f);
        colormaptype = fgetc(f);
        imagetype = fgetc(f);
        // skip the colormap
        for (int i = 0; i < 5; i++) fgetc(f);
        xstart = fgetc(f) + (fgetc(f) << 8);
        ystart = fgetc(f) + (fgetc(f) << 8);
        width  = fgetc(f) + (fgetc(f) << 8);
        height = fgetc(f) + (fgetc(f) << 8);
        bits = fgetc(f);
        descriptor = fgetc(f);
        
        // skip the ident stuff
        for (int i = 0; i < identsize; i++) fgetc(f);

        // check the colormaptype
        assert(colormaptype == 0, "ImageStack can't read tgas with a color map");

        int channels = 0;
        bool rle = false;


        switch(imagetype) {
        case 2: // rgb
            channels = 3;
            rle = false;
            break;
        case 3: // gray
            channels = 1;
            rle = false;
            break;
        case 10: // rgb rle
            channels = 3;
            rle = true;
            break;
        case 11: // gray rle
            channels = 1;
            rle = true;          
            break;
        default:
            panic("ImageStack can't load this type of tga (type %i)\n", imagetype);
        }

        // check for an alpha channel
        if (bits == 32 && channels == 3) channels = 4;

        assert(bits == 8 * channels, "ImageStack only supports 8 bits per channel tgas (this one has %i bits for %i channels)\n", bits, channels);

        Image im(width, height, 1, channels);

        bool vflip = true; //!(descriptor & 0x10);


        float *maxPtr = im(0, 0, 1);
        if (!rle && channels == 1) {
            for (float *ptr = im(0, 0, 0); ptr < maxPtr;) {
                *ptr++ = LDRtoHDR(fgetc(f));
            }
        } else if (!rle && channels == 3) {
            for (float *ptr = im(0, 0, 0); ptr < maxPtr;) {
                ptr[2] = LDRtoHDR(fgetc(f));
                ptr[1] = LDRtoHDR(fgetc(f));
                ptr[0] = LDRtoHDR(fgetc(f));
                ptr += 3;
            }
        } else if (!rle && channels == 4) {
            for (float *ptr = im(0, 0, 0); ptr < maxPtr;) {
                ptr[2] = LDRtoHDR(fgetc(f));
                ptr[1] = LDRtoHDR(fgetc(f));
                ptr[0] = LDRtoHDR(fgetc(f));
                ptr[3] = LDRtoHDR(fgetc(f));
                ptr += 4;
            }
        } else if (rle && channels == 1) {
            for (float *ptr = im(0, 0, 0); ptr < maxPtr;) {
                unsigned char ch = fgetc(f);
                unsigned char runlength = ch & 0x7f;

                if (ch & 0x80) { // compressed
                    float val = LDRtoHDR(fgetc(f));
                    for (int j = 0; j < runlength; j++) {
                        *ptr++ = val;
                    }
                } else { // normal
                    for (int j = 0; j < runlength; j++) {
                        *ptr++ = LDRtoHDR(fgetc(f));
                    }
                }
            }
        } else if (rle && channels == 3) {
            for (float *ptr = im(0, 0, 0); ptr < maxPtr;) {
                unsigned char ch = fgetc(f);
                unsigned char runlength = ch & 0x7f;

                if (ch & 0x80) { // compressed
                    float b = LDRtoHDR(fgetc(f));
                    float g = LDRtoHDR(fgetc(f));
                    float r = LDRtoHDR(fgetc(f));
                    for (int j = 0; j < runlength; j++) {
                        *ptr++ = r;
                        *ptr++ = g;
                        *ptr++ = b;
                    }
                } else { // normal
                    for (int j = 0; j < runlength; j++) {
                        ptr[2] = LDRtoHDR(fgetc(f));
                        ptr[1] = LDRtoHDR(fgetc(f));
                        ptr[0] = LDRtoHDR(fgetc(f));
                        ptr += 3;
                    }
                }
            }
        } else if (rle && channels == 4) {
            for (float *ptr = im(0, 0, 0); ptr < maxPtr;) {
                unsigned char ch = fgetc(f);
                unsigned char runlength = ch & 0x7f;

                if (ch & 0x80) { // compressed
                    float b = LDRtoHDR(fgetc(f));
                    float g = LDRtoHDR(fgetc(f));
                    float r = LDRtoHDR(fgetc(f));
                    float a = LDRtoHDR(fgetc(f));
                    for (int j = 0; j < runlength; j++) {
                        *ptr++ = r;
                        *ptr++ = g;
                        *ptr++ = b;
                        *ptr++ = a;
                    }
                } else { // normal
                    for (int j = 0; j < runlength; j++) {
                        ptr[2] = LDRtoHDR(fgetc(f));
                        ptr[1] = LDRtoHDR(fgetc(f));
                        ptr[0] = LDRtoHDR(fgetc(f));
                        ptr[3] = LDRtoHDR(fgetc(f));
                        ptr += 4;
                    }
                }
            }
        }


        fclose(f);

        
        if (vflip) {
            float  *tmp = new float[width*channels];
            for (int y = 0; y < height/2; y++) {
                memcpy(tmp, im(0, y, 0), width*channels*sizeof(float));
                memcpy(im(0, y, 0), im(0, height-y-1, 0), width*channels*sizeof(float));
                memcpy(im(0, height-y-1, 0), tmp, width*channels*sizeof(float));
            }
            delete[] tmp;
        }

        return im;
    }

    void save(Window im, string filename) {
        FILE *f = fopen(filename.c_str(), "wb");
        assert(f, "Could not open file %s\n", filename.c_str());
        assert(im.frames == 1, "can only save single frame tgas\n");
        assert(im.channels == 4 || im.channels == 3 || im.channels == 1, "can only save tgas with one, three, or four channels\n");

        fputc(0, f); // identsize
        fputc(0, f); // colormaptype
        fputc(im.channels == 1 ? 3 : 2, f); // gray or rgb
        fputc(0, f); // colormap stuff
        fputc(0, f); 
        fputc(0, f); 
        fputc(0, f); 
        fputc(0, f); 
        fputc(0, f); fputc(0, f); // x origin
        fputc(0, f); fputc(0, f); // y origin
        fputc(im.width & 255, f); fputc((im.width >> 8) & 255, f); // width
        fputc(im.height & 255, f); fputc((im.height >> 8) & 255, f); // height
        fputc(im.channels * 8, f); // bits
        fputc(0, f); // descriptor

        if (im.channels == 1) {
            for (int y = im.height-1; y>=0; y--) {
                float *ptr = im(0, y, 0);
                for (int i = 0; i < im.width; i++) {
                    fputc(HDRtoLDR(*ptr++), f);
                }
            }
        } else if (im.channels == 3) {
            for (int y = im.height-1; y>=0; y--) {
                float *ptr = im(0, y, 0);
                for (int i = 0; i < im.width; i++) {
                    fputc(HDRtoLDR(ptr[2]), f);
                    fputc(HDRtoLDR(ptr[1]), f);
                    fputc(HDRtoLDR(ptr[0]), f);
                    ptr += 3;
                }
            }
        } else if (im.channels == 4) {
            for (int y = im.height-1; y>=0; y--) {
                float *ptr = im(0, y, 0);
                for (int i = 0; i < im.width; i++) {
                    fputc(HDRtoLDR(ptr[2]), f);
                    fputc(HDRtoLDR(ptr[1]), f);
                    fputc(HDRtoLDR(ptr[0]), f);
                    fputc(HDRtoLDR(ptr[3]), f);
                    ptr += 4;
                }
            }
        }


        fclose(f);
    }
}
#include "footer.h"
