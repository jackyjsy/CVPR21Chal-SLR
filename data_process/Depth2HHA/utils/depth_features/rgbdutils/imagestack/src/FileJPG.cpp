#include "main.h"
#include "File.h"

#ifdef NO_JPEG

#include "header.h"
namespace FileJPG {
#include "FileNotImplemented.h"
}
#include "footer.h"

#else

extern "C" {
#include <jpeglib.h>
}

#include "header.h"
namespace FileJPG {

    void help() {
        printf(".jpg (or .jpeg) files. When saving, an optional second arguments specifies\n"
               "the quality. This defaults to 90. A jpeg image always has a single frame,\n"
               "and may have either one or three channels.\n");
    }

    void save(Window im, string filename, int quality) {
        assert(im.channels == 1 || im.channels == 3, "Can only save jpg images with 1 or 3 channels\n");
        assert(im.frames == 1, "Can't save multiframe jpg images\n");
        assert(quality > 0 && quality <= 100, "jpeg quality must lie between 1 and 100\n");        

        struct jpeg_compress_struct cinfo;
        struct jpeg_error_mgr jerr;

        FILE *f = fopen(filename.c_str(), "wb");
        assert(f, "Could not open file %s\n", filename.c_str());

        cinfo.err = jpeg_std_error(&jerr);
        jpeg_create_compress(&cinfo);
        jpeg_stdio_dest(&cinfo, f);

        cinfo.image_width = im.width;
        cinfo.image_height = im.height;
        cinfo.input_components = im.channels;
        if (im.channels == 3) {
            cinfo.in_color_space = JCS_RGB;  
        } else { // channels must be 1
            cinfo.in_color_space = JCS_GRAYSCALE;  
        }

        jpeg_set_defaults(&cinfo);
        jpeg_set_quality(&cinfo, quality, TRUE);

        jpeg_start_compress(&cinfo, TRUE);

        JSAMPLE *row = new JSAMPLE[im.width * im.channels];

        while (cinfo.next_scanline < cinfo.image_height) {
            // convert the row
            JSAMPLE *dstPtr = row;
            for (int x = 0; x < im.width; x++) {
                for (int c = 0; c < im.channels; c++) {
                    *dstPtr++ = (JSAMPLE)(HDRtoLDR(im(x, cinfo.next_scanline)[c]));
                }
            }
            jpeg_write_scanlines(&cinfo, &row, 1);
        }

        jpeg_finish_compress(&cinfo);
        fclose(f);

        // clean up
        delete[] row;
        jpeg_destroy_compress(&cinfo);

    }



    Image load(string filename) {

        struct jpeg_decompress_struct cinfo;
        struct jpeg_error_mgr jerr;

        FILE *f = fopen(filename.c_str(), "rb");
        assert(f, "Could not open file %s\n", filename.c_str());

        cinfo.err = jpeg_std_error(&jerr);
        jpeg_create_decompress(&cinfo);
        jpeg_stdio_src(&cinfo, f);

        jpeg_read_header(&cinfo, TRUE);
        jpeg_start_decompress(&cinfo);

        Image im(cinfo.output_width, cinfo.output_height, 1, cinfo.output_components);
        JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, im.width * im.channels, 1);

        while (cinfo.output_scanline < cinfo.output_height) {
            jpeg_read_scanlines(&cinfo, buffer, 1);
            JSAMPLE *srcPtr = buffer[0];
            for (int x = 0; x < im.width; x++) {
                for (int c = 0; c < im.channels; c++) {
                    im(x, cinfo.output_scanline-1)[c] = LDRtoHDR(*srcPtr++);
                }
            }
        }

        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);

        fclose(f);

        return im;
    }
}
#include "footer.h"
#endif
