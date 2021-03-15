#include "main.h"
#include "Projection.h"
#include "Arithmetic.h"
#include "header.h"

void Sinugram::help() {
}

void Sinugram::parse(vector<string> args) {
    assert(args.size() == 1, "-sinugram takes one argument");
    Image im = apply(stack(0), readInt(args[0]));
    pop();
    push(im);
}

Image Sinugram::apply(Window im, int directions) {
    int outWidth = (int)(ceil(sqrtf(im.width * im.width + im.height * im.height)));

    Image out(outWidth, directions, im.frames, im.channels);
    Image weight(outWidth, directions, im.frames, 1);

    for (int t = 0; t < out.frames; t++) {
        for (int d = 0; d < directions; d++) {
            float theta = (float)((d * M_PI) / directions);
            float dx = cos(theta);
            float dy = sin(theta);
            for (int y = 0; y < im.height; y++) {
                for (int x = 0; x < im.width; x++) {
                    // calculate the distance from this pixel to 
                    // the ray passing through the center at angle theta
                    float distance = (x - im.width * 0.5f) * dy + (y - im.height * 0.5f) * -dx;
                    float outX = distance + outWidth * 0.5f;
                    float outXf = outX - (int)outX;
                    for (int c = 0; c < im.channels; c++) {
                        if (outX > 0) {
                            out((int)outX, d, t)[c] += (1-outXf) * im(x, y, t)[c];
                            weight((int)outX, d, t)[c] += (1-outXf);
                        }
                        if (outX < out.width-1) {
                            out((int)outX+1, d, t)[c] += outXf * im(x, y, t)[c];
                            weight((int)outX+1, d, t)[c] += outXf;
                        }
                    }
                }
            }
        }
    }

    Divide::apply(out, weight);
    return out;
}

#include "footer.h"
