#include "main.h"
#include "File.h"
#include "header.h"

namespace FileCSV {
    void help() {
        pprintf(".csv files. These contain comma-separated floating point values in"
                " text. Each scanline of the image corresponds to a line in the file. x"
                " and c are thus conflated, as are y and t. When loading csv files,"
                " ImageStack assumes 1 channel and 1 frame.\n");
    }


    Image load(string filename) {
        // calculate the number of rows and columns in the file
        FILE *f = fopen(filename.c_str(), "r");

        // how many commas in the first line?
        int width = 1;
        int c, last;
        do {
            c = fgetc(f);
            if (c == ',') width++;
        } while (c != '\n' && c != EOF);

        // how many lines in the file?
        int height = 1;
        do {
            last = c;
            c = fgetc(f);
            if (c == '\n' && last != '\n') height++;
        } while (c != EOF);

        printf("%d %d\n", width, height);

        // go back to the start and start reading data
        fseek(f, 0, SEEK_SET);

        Image out(width, height, 1, 1);
        
        for (int y = 0; y < height; y++) {
            float *outPtr = out(0, y, 0);
            for (int x = 0; x < width-1; x++) {
                assert(fscanf(f, "%f,", outPtr) == 1, "Failed to parse file\n");
                printf("%f, ", *outPtr);
                outPtr++;
            }
            assert(fscanf(f, "%f", outPtr) == 1, "Failed to parse file\n");
            printf("%f\n", *outPtr);
        }        

        fclose(f);

        return out;
    }

    void save(Window im, string filename) {
        FILE *f = fopen(filename.c_str(), "w");

        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {
                float *imPtr = im(0, y, t);
                for (int x = 0; x < im.width * im.channels-1; x++) {
                    fprintf(f, "%10.10f, ", *imPtr);
                    imPtr++;
                }
                fprintf(f, "%10.10f\n", *imPtr);
            }
        }

        fclose(f);
    }
}
#include "footer.h"
