#include "main.h"
#include "File.h"

#ifdef NO_OPENEXR
#include "header.h"
namespace FileEXR {
    #include "FileNotImplemented.h"
}
#include "footer.h"
#else 

#include <ImfRgbaFile.h>
#include <ImfArray.h>
#include <ImfChannelList.h>

#include "header.h"
namespace FileEXR {

    void help() {
        printf(".exr files. These store high dynamic range data as 16 bit floats. They must\n"
               "have one frame, and one to four channels.  When saving, the optional second\n"
               "argument represents the compression type, and must be one of none, rle,\n"
               "zips, zip, piz, or pxr24. piz is the default, all but pxr24 are lossless.\n");
    }
  
    Image load(string filename) {
        Imf::Array2D<Imf::Rgba> pixels;
        Imf::RgbaInputFile file(filename.c_str());
        Imath::Box2i dw = file.dataWindow();

        assert(file.isComplete(), "Failed to read file %s\n", filename.c_str());

        unsigned int numChannels = 0;
        // Only going to accept a subset of all possible channel possibilities
        if (file.channels() == Imf::WRITE_RGBA ) {
            numChannels = 4;
        } else if (file.channels() == Imf::WRITE_RGB) {
            numChannels = 3;
        } else if (file.channels() == Imf::WRITE_YA) {
            numChannels = 2;
        } else if (file.channels() == Imf::WRITE_Y) {
            numChannels = 1;
        } else {
            panic("ImageStack cannot load EXR files that are not either RGBA, RGB, YA, or Y");
        }

        int width  = dw.max.x - dw.min.x + 1;
        int height = dw.max.y - dw.min.y + 1;
        pixels.resizeErase(height, width);
        
        file.setFrameBuffer(&pixels[0][0] - dw.min.x - dw.min.y * width, 1, width);
        file.readPixels(dw.min.y, dw.max.y);
        
        Image im(width, height, 1, numChannels);

        float *ptr = im.data;        
        switch(numChannels) {
        case 1:  
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    Imf::Rgba &p = pixels[y][x]; 
                    // Y-only EXR image is converted to R=Y,G=Y,B=Y when read as
                    // an RGBA image
                    *ptr++ = (float)(p.r);
                }
            }
            break;
        case 2: 
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    Imf::Rgba &p = pixels[y][x]; 
                    // YA-only EXR image is converted to R=Y,G=Y,B=Y when read as
                    // an RGBA image
                    *ptr++ = (float)(p.r);
                    *ptr++ = (float)(p.a);
                }
            }
            break;
        case 3: 
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    Imf::Rgba &p = pixels[y][x]; 
                    *ptr++ = (float)(p.r);
                    *ptr++ = (float)(p.g);
                    *ptr++ = (float)(p.b);
                }
            }
            break;
        case 4: 
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    Imf::Rgba &p = pixels[y][x]; 
                    *ptr++ = (float)(p.r);
                    *ptr++ = (float)(p.g);
                    *ptr++ = (float)(p.b);
                    *ptr++ = (float)(p.a);
                }
            }
            break;
        }
        return im;
    }

    void save(Window im, string filename, string compression = "piz") {
        Imf::RgbaChannels flags=Imf::WRITE_C; // not going to be used
        int width = im.width;
        int height = im.height;
        
        assert(im.frames == 1, "Can't save a multi-frame EXR image\n");
        
        if (im.channels == 1) flags = Imf::WRITE_Y;
        else if (im.channels == 2) flags = Imf::WRITE_YA;
        else if (im.channels == 3) flags = Imf::WRITE_RGB;
        else if (im.channels == 4) flags = Imf::WRITE_RGBA;
        else {
            panic("Imagestack can't write exr files that have other than 1, 2, 3, or 4 channels\n");
        }

        Imf::Compression comp;
        if (compression == "none") {
            comp = Imf::NO_COMPRESSION;
        } else if (compression == "rle") {
            comp = Imf::RLE_COMPRESSION;
        } else if (compression == "zips") {
            comp = Imf::ZIPS_COMPRESSION;
        } else if (compression == "zip") {
            comp = Imf::ZIP_COMPRESSION;
        } else if (compression == "piz") {
            comp = Imf::PIZ_COMPRESSION;
        } else if (compression == "pxr24") {
            comp = Imf::PIZ_COMPRESSION;
        } else panic ("saveEXR: Unknown compression type %s!\n", compression.c_str());
        
        Imf::Array2D<Imf::Rgba> pixels;
        pixels.resizeErase(height, width);
        
        if (im.channels == 1) {
            float *ptr = im(0, 0);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    Imf::Rgba &p = pixels[y][x]; 
                    p.r = p.g = p.b = *ptr++;
                    p.a = 1;
                }
            }
        } else if (im.channels == 2) {
            float *ptr = im(0, 0);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    Imf::Rgba &p = pixels[y][x]; 
                    p.r = p.g = p.b = *ptr++;
                    p.a = *ptr++;
                }
            }
        } else if (im.channels == 3) {
            float *ptr = im(0, 0);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    Imf::Rgba &p = pixels[y][x]; 
                    p.r = *ptr++;
                    p.g = *ptr++;
                    p.b = *ptr++;
                    p.a = 1;
                }
            }
        } else if (im.channels == 4) {
            float *ptr = im(0,0);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    Imf::Rgba &p = pixels[y][x]; 
                    p.r = *ptr++;
                    p.g = *ptr++;
                    p.b = *ptr++;
                    p.a = *ptr++;
                }
            }        
        } else {
            panic("saveEXR: This case shouldn't happen.");
        }

            
        
        Imf::RgbaOutputFile file(filename.c_str(), width, height, flags,
                                 1, Imath::V2f(0,0), 1, Imf::INCREASING_Y,
                                 comp);
        file.setFrameBuffer(&pixels[0][0], 1, width);
        file.writePixels(height);
    }

}

#include "footer.h"
#endif

