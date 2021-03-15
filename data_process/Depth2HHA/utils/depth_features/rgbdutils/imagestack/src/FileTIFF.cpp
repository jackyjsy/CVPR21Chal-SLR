#include "main.h"
#include "File.h"

#ifndef NO_TIFF
#ifndef NO_OPENEXR
#include <half.h>
#endif
#endif

#include "header.h"

#ifdef NO_TIFF

namespace FileTIFF {
#include "FileNotImplemented.h"
}

#else
#include "Arithmetic.h"

#include <limits>

namespace FileTIFF {
    
    #include <tiffio.h>

    void help() {
        printf(".tiff (or .tif or .meg) files. When saving, an optional second argument\n"
               "specifies the format. This may be any of int8, uint8, int16, uint16, int32,\n"
               "uint32, float16, float32, float64, or correspondingly char, unsigned char,\n"
               "short, unsigned short, int, unsigned int, half, float, or double. The default\n"
               "is uint16.\n");
    }


    template<typename T>
    void readTiff(Image &im, TIFF *tiff, unsigned int divisor) {
        T *buffer = new T[im.channels * im.width];

        float multiplier = 1.0f / divisor;

        for (int y = 0; y < im.height; y++) {
            assert(TIFFReadScanline(tiff, buffer, y, 1) != -1, 
                   "Failed reading scanline\n");
            for (int x = 0; x < im.width; x++) {
                for (int c = 0; c < im.channels; c++) {
                    im(x, y)[c] = ((float)buffer[x * im.channels + c]) * multiplier;
                }
            }
        }        

        delete[] buffer;
    }

    Image load(string filename) {
        TIFF *tiff = TIFFOpen(filename.c_str(), "r");

        assert(tiff, "Could not open file %s\n", filename.c_str());

        // Get basic information from TIFF header
        int w;
        assert(TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &w), 
               "Width not set in TIFF\n");
        int h;
        assert(TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &h), 
               "Height not set in tiff\n");
        unsigned short c;
        assert(TIFFGetField(tiff, TIFFTAG_SAMPLESPERPIXEL, &c),
               "Number of channels not set in tiff\n");

        unsigned short bitsPerSample;
        assert(TIFFGetField(tiff, TIFFTAG_BITSPERSAMPLE, &bitsPerSample),
               "Bits per sample not set in TIFF\n");

        unsigned short sampleFormat;
        if (!TIFFGetField(tiff, TIFFTAG_SAMPLEFORMAT, &sampleFormat)) {
            //printf("WARNING: couldn't find sample format in tiff, assuming %i bit unsigned integers\n", bitsPerSample);
            sampleFormat = SAMPLEFORMAT_UINT;
        }

        Image im(w, h, 1, c);
        int bytesPerSample = bitsPerSample / 8;

        assert(im.channels * im.width * bytesPerSample == TIFFScanlineSize(tiff), 
               "Unsupported scanline format in TIFF file, might be stored in tiles or strips.\n");

        if (bytesPerSample == 1 && sampleFormat == SAMPLEFORMAT_UINT) {        
            readTiff<unsigned char>(im, tiff, 0x000000ff);
        } else if (bytesPerSample == 1 && sampleFormat == SAMPLEFORMAT_INT) {
            readTiff<char>(im, tiff, 0x000000ff);
        } else if (bytesPerSample == 2 && sampleFormat == SAMPLEFORMAT_UINT) {
            readTiff<unsigned short>(im, tiff, 0x0000ffff);
        } else if (bytesPerSample == 2 && sampleFormat == SAMPLEFORMAT_INT) {
            readTiff<short>(im, tiff, 0x0000ffff);
        #ifndef NO_OPENEXR
        } else if (bytesPerSample == 2 && sampleFormat == SAMPLEFORMAT_IEEEFP) {
            readTiff<half>(im, tiff, 1);
        #endif
        } else if (bytesPerSample == 4 && sampleFormat == SAMPLEFORMAT_UINT) {
            readTiff<unsigned int>(im, tiff, 0xffffffff);
        } else if (bytesPerSample == 4 && sampleFormat == SAMPLEFORMAT_INT) {
            readTiff<int>(im, tiff, 0xffffffff);
        } else if (bytesPerSample == 4 && sampleFormat == SAMPLEFORMAT_IEEEFP) {
            readTiff<float>(im, tiff, 1);
        } else if (bytesPerSample == 8 && sampleFormat == SAMPLEFORMAT_IEEEFP) {
            readTiff<double>(im, tiff, 1);
        } else if (sampleFormat == SAMPLEFORMAT_UINT || sampleFormat == SAMPLEFORMAT_INT) {
            panic("%i bytes per sample for integers unsupported\n", bytesPerSample);
        } else if (sampleFormat == SAMPLEFORMAT_IEEEFP) {
            panic("%i bytes per sample for floats unsupported\n", bytesPerSample);
        } else {
            panic("Sample format unsupported (not int, unsigned int, or float)\n");
        }

        TIFFClose(tiff);

        return im;
    }


    template<typename T>
    void writeTiff(Window im, TIFF *tiff, unsigned int multiplier) {


        double minval = (double)std::numeric_limits<T>::min();
        double maxval = (double)std::numeric_limits<T>::max();

        bool clamped = false;

        T *buffer = new T[im.width * im.channels];
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                for (int c = 0; c < im.channels; c++) {
                    double out = im(x, y)[c] * multiplier;
                    if (out < minval) {clamped = true; out = minval;}
                    if (out > maxval) {clamped = true; out = maxval;}
                    buffer[x * im.channels + c] = (T)(out);
                }
            }
            TIFFWriteScanline(tiff, buffer, y, 1);
        }        

        delete[] buffer;

        if (clamped) printf("WARNING: Data exceeded the range [0, 1], so was clamped on writing.\n");
    }

    void save(Window im, string filename, string type) {
        // Open 16-bit TIFF file for writing
        TIFF *tiff = TIFFOpen(filename.c_str(), "w");
        assert(tiff, "Could not open file %s\n", filename.c_str());

        if (type == "") {
            type = "uint16";
            printf("WARNING: type not specified, assuming 16 bit unsigned int\n");
        }

        assert(im.frames == 1, "Can only save single frame tiffs\n");
        
        TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, im.channels);
        TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, im.width);
        TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, im.height);

        if (im.channels == 1) TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK); // grayscale, black is 0
        else if (im.channels == 3) TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
        else {
            printf("WARNING: Image is neither 1 channel nor 3 channels, so cannot set a valid photometric interpretation.\n");
        }
        TIFFSetField(tiff, TIFFTAG_ROWSPERSTRIP, 1L);
        TIFFSetField(tiff, TIFFTAG_XRESOLUTION, 1.0);
        TIFFSetField(tiff, TIFFTAG_YRESOLUTION, 1.0);
        TIFFSetField(tiff, TIFFTAG_RESOLUTIONUNIT, 1);
        TIFFSetField(tiff, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
        TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(tiff, TIFFTAG_ORIENTATION, (int)ORIENTATION_TOPLEFT);

        if (type == "int8" || type == "char") {
            TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, 8);
            TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_INT);
            writeTiff<char>(im, tiff, 0x000000ff);
        } else if (type == "uint8" || type == "unsigned char") {
            TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, 8);
            TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
            writeTiff<unsigned char>(im, tiff, 0x000000ff);
        } else if (type == "int16" || type == "short") {
            TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, 16);
            TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_INT);
            writeTiff<short>(im, tiff, 0x0000ffff);
        } else if (type == "uint16" || type == "unsigned short") {
            TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, 16);
            TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
            writeTiff<unsigned short>(im, tiff, 0x0000ffff);
        #ifndef NO_OPENEXR
        } else if (type == "float16" || type == "half") {
            TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, 16);
            TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
            writeTiff<half>(im, tiff, 1);
        #endif
        } else if (type == "int32" || type == "int") {
            TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, 32);
            TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_INT);
            writeTiff<int>(im, tiff, 0xffffffff);
        } else if (type == "uint32" || type == "unsigned int") {
            TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, 32);
            TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
            writeTiff<unsigned int>(im, tiff, 0xffffffff);
        } else if (type == "float32" || type == "float") {
            TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, 32);
            TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
            writeTiff<float>(im, tiff, 1);
        } else if (type == "float64" || type == "double") {
            TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, 64);
            TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
            writeTiff<double>(im, tiff, 1);
        } else {
            panic("Unknown type %s\n", type.c_str());
        }

        // Close 16-bit TIFF file
        TIFFClose(tiff);        
    }


}

#endif
#include "footer.h"
