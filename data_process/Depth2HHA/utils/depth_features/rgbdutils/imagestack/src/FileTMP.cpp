#include "main.h"
#include "File.h"
#include "header.h"

namespace FileTMP {
    enum TypeCode {FLOAT32 = 0, FLOAT64, UINT8, INT8, UINT16, INT16, UINT32, INT32, UINT64, INT64};

    void help() {
        pprintf(".tmp files. This format is used to save temporary image data, and to"
                " interoperate with other programs that can load or save raw binary"
                " data. The format supports any number of frames and channels."
                " A .tmp file starts with a header that containining five 32-bit"
                " integer values which represents frames, width, height, channels and"
                " type. Image data follows.\n"
                "types:\n"
                " 0: 32 bit floats (the default format, which matches the internal format)\n"
                " 1: 64 bit doubles\n"
                " 2: 8 bit unsigned integers\n"
                " 3: 8 bit signed integers\n"
                " 4: 16 bit unsigned integers\n"
                " 5: 16 bit signed integers\n"
                " 6: 32 bit unsigned integers\n"
                " 7: 32 bit signed integers\n"
                " 8: 64 bit unsigned integers\n"
                " 9: 64 bit signed integers\n"
                "\n"
                "When saving, an optional second argument specifies the format. This"
                " may be any of int8, uint8, int16, uint16, int32, uint32, int64,"
                " uint64, float32, float64, or correspondingly char, unsigned"
                " char, short, unsigned short, int, unsigned int, float, or"
                " double. The default is float32.\n");
    }

    template<typename T>
    void saveData(FILE *f, Window im) {

        vector<T> buf(im.width*im.channels);
        
        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {
                float *srcPtr = im(0, y, t);
                T *dstPtr = &buf[0];
                for (int j = 0; j < im.width*im.channels; j++) {
                    *dstPtr++ = (T)(*srcPtr++);
                }
                fwrite(&buf[0], sizeof(T), im.width*im.channels, f);
            }
        }
    }

    template<>
    void saveData<float>(FILE *f, Window im) {

        vector<float> buf(im.width*im.channels);
        
        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {
                fwrite(im(0, y, t), sizeof(float), im.width*im.channels, f);
            }
        }
    }


    void save(Window im, string filename, string type) {
        FILE *f = fopen(filename.c_str(), "wb");
        assert(f, "Could not write output file %s\n", filename.c_str());
        // write the dimensions
        fwrite(&im.frames, sizeof(int), 1, f);
        fwrite(&im.width, sizeof(int), 1, f);
        fwrite(&im.height, sizeof(int), 1, f);
        fwrite(&im.channels, sizeof(int), 1, f);        

        int typeCode;

        if (type == "float" || type == "float32") {
            typeCode = FLOAT32;
            fwrite(&typeCode, sizeof(int), 1, f);
            saveData<float>(f, im);
        } else if (type == "double" || type == "float64") {
            typeCode = FLOAT64;
            fwrite(&typeCode, sizeof(int), 1, f);
            saveData<double>(f, im);
        } else if (type == "uint8" || type == "unsigned char") {
            typeCode = UINT8;
            fwrite(&typeCode, sizeof(int), 1, f);
            saveData<unsigned char>(f, im);
        } else if (type == "int8" || type == "char") {
            typeCode = INT8;
            fwrite(&typeCode, sizeof(int), 1, f);
            saveData<signed char>(f, im);
        } else if (type == "uint16" || type == "unsigned short") {
            typeCode = UINT16;
            fwrite(&typeCode, sizeof(int), 1, f);
            saveData<unsigned short>(f, im);
        } else if (type == "int16" || type == "short") {
            typeCode = INT16;
            fwrite(&typeCode, sizeof(int), 1, f);
            saveData<signed short>(f, im);
        } else if (type == "uint32" || type == "unsigned int") {
            typeCode = UINT32;
            fwrite(&typeCode, sizeof(int), 1, f);
            saveData<unsigned long>(f, im);
        } else if (type == "int32" || type == "int") {
            typeCode = INT32;
            fwrite(&typeCode, sizeof(int), 1, f);
            saveData<signed long>(f, im);
        } else if (type == "uint64") {
            typeCode = UINT64;
            fwrite(&typeCode, sizeof(int), 1, f);
            saveData<unsigned long long>(f, im);
        } else if (type == "int64") {
            typeCode = INT64;
            fwrite(&typeCode, sizeof(int), 1, f);
            saveData<signed long long>(f, im);
        }
        fclose(f);
    }

    template<typename T>
    Image loadData(FILE *f, int frames, int width, int height, int channels) {
        int size = width * height * channels * frames;

        Image im(width, height, frames, channels);

        const size_t chunkSize = 1<<12;

        T buf[chunkSize];

        float *dstPtr = im(0, 0, 0);
        float *endPtr = im(0, 0, im.frames);
        
        T *srcPtr = &buf[0];
        for (int i = 0; i < size; i += chunkSize) {
            srcPtr = &buf[0];
            size_t sizeLeft = size - i - 1;
            size_t expectedSize = (sizeLeft < chunkSize) ? sizeLeft : chunkSize;
            assert(fread(srcPtr, sizeof(T), expectedSize, f) == expectedSize,
                   "Unexpected end of file\n");
            for (size_t j = 0; j < chunkSize && dstPtr != endPtr; j++) {
                *dstPtr++ = (float)(*srcPtr++);
            }
        }
        
        return im;
    }

    template<>
    Image loadData<float>(FILE *f, int frames, int width, int height, int channels) {
        size_t size = width * height * channels * frames;

        Image im(width, height, frames, channels);
        
        assert(fread(im(0, 0, 0), sizeof(float), size, f) == size,
               "Unexpected end of file\n");
        
        return im;
    }

    Image load(string filename) {
        FILE *file = fopen(filename.c_str(), "rb");
        assert(file, "Could not open file %s\n", filename.c_str());
        
        // get the dimensions
        struct header_t {
            int frames, width, height, channels, typeCode;
        } h;
        assert(fread(&h, sizeof(int), 5, file) == 5,
               "File ended before end of header\n");

        Image im;

        if (h.typeCode == FLOAT32) {
            im = loadData<float>(file, h.frames, h.width, h.height, h.channels);                                                                                    
        } else if (h.typeCode == FLOAT64) {
            im = loadData<double>(file, h.frames, h.width, h.height, h.channels);
        } else if (h.typeCode == UINT8) {
            im = loadData<unsigned char>(file, h.frames, h.width, h.height, h.channels);            
        } else if (h.typeCode == INT8) {
            im = loadData<signed char>(file, h.frames, h.width, h.height, h.channels);            
        } else if (h.typeCode == UINT16) {
            im = loadData<unsigned short>(file, h.frames, h.width, h.height, h.channels);            
        } else if (h.typeCode == INT16) {
            im = loadData<signed short>(file, h.frames, h.width, h.height, h.channels);            
        } else if (h.typeCode == UINT32) {
            im = loadData<unsigned long>(file, h.frames, h.width, h.height, h.channels);            
        } else if (h.typeCode == INT32) {
            im = loadData<signed long>(file, h.frames, h.width, h.height, h.channels);            
        } else if (h.typeCode == UINT64) {
            im = loadData<unsigned long long>(file, h.frames, h.width, h.height, h.channels);            
        } else if (h.typeCode == INT64) {
            im = loadData<signed long long>(file, h.frames, h.width, h.height, h.channels);            
        } else {
            printf("Unknown type code %d. Possibly trying to load an old-style tmp file.\n", h.typeCode);
            fseek(file, 16, SEEK_SET);
            im = loadData<float>(file, h.frames, h.width, h.height, h.channels);
        }

        fclose(file);

        return im;
    }
}
#include "footer.h"
