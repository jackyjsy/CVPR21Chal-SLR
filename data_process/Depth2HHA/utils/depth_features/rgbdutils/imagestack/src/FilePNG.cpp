#include "main.h"
#include "File.h"
#include "header.h"

#ifdef NO_PNG
namespace FilePNG {
#include "FileNotImplemented.h"
}
#else

namespace FilePNG {

#define PNG_DEBUG 3
#include <png.h>

    void help() {
        printf(".png files. These have a bit depth of 8, and may have 1-4 channels. They may\n"
               "only have 1 frame.\n");
    }

    Image load(string filename) {
        png_byte header[8];        // 8 is the maximum size that can be checked
        png_structp png_ptr;
        png_infop info_ptr;
        int number_of_passes;
        png_bytep * row_pointers;
    
        /* open file and test for it being a png */
        FILE *f = fopen(filename.c_str(), "rb");
        assert(f, "File %s could not be opened for reading\n", filename.c_str());
        assert(fread(header, 1, 8, f) == 8, "File ended before end of header\n");
        assert(!png_sig_cmp(header, 0, 8), "File %s is not recognized as a PNG file\n", filename.c_str());
    
        /* initialize stuff */
        png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    
        assert(png_ptr, "[read_png_file] png_create_read_struct failed\n");
    
        info_ptr = png_create_info_struct(png_ptr);
        assert(info_ptr, "[read_png_file] png_create_info_struct failed\n");
    
        assert(!setjmp(png_jmpbuf(png_ptr)), "[read_png_file] Error during init_io\n");
    
        png_init_io(png_ptr, f);
        png_set_sig_bytes(png_ptr, 8);
    
        png_read_info(png_ptr, info_ptr);
    
        int width = png_get_image_width(png_ptr, info_ptr);
        int height = png_get_image_height(png_ptr, info_ptr);
        int channels = png_get_channels(png_ptr, info_ptr);
        int bit_depth = png_get_bit_depth(png_ptr, info_ptr);

        // Expand low-bpp images to have only 1 pixel per byte (As opposed to tight packing)
        if (bit_depth < 8)
            png_set_packing(png_ptr);

        Image im(width, height, 1, channels);

        number_of_passes = png_set_interlace_handling(png_ptr);
        png_read_update_info(png_ptr, info_ptr);
    
        // read the file
        assert(!setjmp(png_jmpbuf(png_ptr)), "[read_png_file] Error during read_image\n");
    
        row_pointers = new png_bytep[im.height];
        for (int y = 0; y < im.height; y++)
            row_pointers[y] = new png_byte[info_ptr->rowbytes];
    
        png_read_image(png_ptr, row_pointers);
    
        fclose(f);
    
        // convert the data to floats
        if (bit_depth <= 8) {
            int bit_scale = 8/bit_depth;
            for (int y = 0; y < im.height; y++) {
                png_bytep srcPtr = row_pointers[y];
                for (int x = 0; x < im.width; x++) {
                    for (int c = 0; c < im.channels; c++) {
                        im(x, y)[c] = LDRtoHDR(bit_scale* (*srcPtr++) );
                    }
                }
            }
        } else if (bit_depth == 16) {
            printf("Reading a 16-bit PNG image (Image may be darker than expected!)\n");
            for (int y = 0; y < im.height; y++) {
                png_bytep srcPtr = row_pointers[y];
                for (int x = 0; x < im.width; x++) {
                    for (int c = 0; c < im.channels; c++) {
                        im(x, y)[c] = LDR16toHDR(*srcPtr);  // Note: Endian issues may be possible here, seems to work in WinXP
                        srcPtr+=2;                                               
                    }
                }
            }
        }
    
        // clean up
        for (int y = 0; y < im.height; y++)
            delete[] row_pointers[y];
        delete[] row_pointers;
    
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);

        return im;
    }
    

    void save(Window im, string filename) {
        png_structp png_ptr;
        png_infop info_ptr;
        png_bytep * row_pointers;
        png_byte color_type;

        assert(im.frames == 1, "Can't save a multi-frame PNG image\n");
        assert(im.channels > 0 && im.channels < 5, 
               "Imagestack can't write PNG files that have other than 1, 2, 3, or 4 channels\n");

        png_byte color_types[4] = {PNG_COLOR_TYPE_GRAY, PNG_COLOR_TYPE_GRAY_ALPHA, 
                                   PNG_COLOR_TYPE_RGB,  PNG_COLOR_TYPE_RGB_ALPHA};
        color_type = color_types[im.channels - 1];
    
        // open file
        FILE *f = fopen(filename.c_str(), "wb");
        assert(f, "[write_png_file] File %s could not be opened for writing\n", filename.c_str());
    
        // initialize stuff
        png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        assert(png_ptr, "[write_png_file] png_create_write_struct failed\n");
    
        info_ptr = png_create_info_struct(png_ptr);
        assert(info_ptr, "[write_png_file] png_create_info_struct failed\n");
    
        assert(!setjmp(png_jmpbuf(png_ptr)), "[write_png_file] Error during init_io\n");
    
        png_init_io(png_ptr, f);
    
        // write header
        assert(!setjmp(png_jmpbuf(png_ptr)), "[write_png_file] Error during writing header\n");
    
        png_set_IHDR(png_ptr, info_ptr, im.width, im.height,
                     8, color_type, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    
        png_write_info(png_ptr, info_ptr);
    
        // convert the floats to bytes
        row_pointers = new png_bytep[im.height];
        for (int y = 0; y < im.height; y++) {
            row_pointers[y] = new png_byte[info_ptr->rowbytes];
            png_bytep dstPtr = row_pointers[y];
            for (int x = 0; x < im.width; x++) {
                for (int c = 0; c < im.channels; c++) {
                    *dstPtr++ = (png_byte)(HDRtoLDR(im(x, y)[c]));
                }
            }
        }
    
        // write data
        assert(!setjmp(png_jmpbuf(png_ptr)), "[write_png_file] Error during writing bytes");
    
        png_write_image(png_ptr, row_pointers);
    
        // finish write
        assert(!setjmp(png_jmpbuf(png_ptr)), "[write_png_file] Error during end of write");
    
        png_write_end(png_ptr, NULL);
    
        // clean up
        for (int y = 0; y < im.height; y++)
            delete[] row_pointers[y];
        delete[] row_pointers;
    
        fclose(f);
    
        png_destroy_write_struct(&png_ptr, &info_ptr);
    }


}

#endif
#include "footer.h"
