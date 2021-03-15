#include "main.h"
#include "File.h"

// first four bytes, should be the same in little endian
#define TAG_FLOAT 202021.25  // check for this when READING the file
#define TAG_STRING "PIEH"    // use this when WRITING the file

#include "header.h"
namespace FileFLO {
    void help() {
        pprintf(".flo files. This format is used for optical flow evaluation. It stores"
                " 2-band float image for horizontal and vertical flow components.\n");
    }

    void save(Window im, string filename) {

        assert(im.channels == 2, "image must have 2 channels", filename.c_str());

        FILE *stream = fopen(filename.c_str(), "wb");
        assert(stream, "Could not open %s", filename.c_str());
        
        // write the header
        fprintf(stream, TAG_STRING);
        if ((int)fwrite(&im.width,  sizeof(int), 1, stream) != 1 ||
            (int)fwrite(&im.height, sizeof(int), 1, stream) != 1)
            panic("problem writing header: %s", filename.c_str());
        
        fwrite(im(0, 0, 0), sizeof(float), im.width * im.height * im.channels, stream);
        fclose(stream);

    }

    Image load(string filename) {
        FILE *stream = fopen(filename.c_str(), "rb");
        assert(stream, "Could not open file %s\n", filename.c_str());

        int width, height;
        float tag;

        if ((int)fread(&tag,    sizeof(float), 1, stream) != 1 ||
            (int)fread(&width,  sizeof(int),   1, stream) != 1 ||
            (int)fread(&height, sizeof(int),   1, stream) != 1)
            panic("ReadFlowFile: problem reading file %s", filename.c_str());

        assert(tag == TAG_FLOAT, "Wrong tag (possibly due to big-endian machine?)");

        // another sanity check to see that integers were read correctly (999999 should do the trick...)
        assert(width > 0 && width < 999999, "illegal width %d", width);
        assert(height > 0 && height < 999999, "illegal height %d", height);
        
        Image out(width, height, 1, 2);

        size_t n = width*height*out.channels;
        assert(fread(out(0, 0, 0), sizeof(float), n, stream) == n,
               "Unexpected end of file\n");

        fclose(stream);

        return out;
    }
}


#include "footer.h"
