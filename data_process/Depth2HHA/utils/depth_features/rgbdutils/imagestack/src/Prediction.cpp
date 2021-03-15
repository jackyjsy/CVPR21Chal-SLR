#include "main.h"
#include "Prediction.h"
#include "Geometry.h"
#include "DFT.h"
#include "header.h"

void Inpaint::help() {
    printf("\n-inpaint takes the image on the top of the stack, and a one channel mask of the\n"
           "same size second on the stack, and diffuses areas of the image where the mask is\n"
           "high into areas of the image where the mask is low. Image pixels with mask of 1\n"
           "are unchanged.\n\n"
           "Usage: ImageStack -push 1 640 480 1 -eval \"(X > 0.5)*(X < 0.6)\" -load in.jpg\n"
           "                  -inpaint -save out.jpg\n\n");
}

void Inpaint::parse(vector<string> args) {
    assert(args.size() == 0, "-inpaint takes no arguments\n");
    Image im = apply(stack(0), stack(1));
    pop();
    push(im);
}

Image Inpaint::apply(Window im, Window mask) {
    assert(im.width == mask.width &&
           im.height == mask.height &&
           im.frames == mask.frames,
           "mask must be the same size as the image\n");
    assert(mask.channels == 1,
           "mask must have one channel\n");

    Image out(im.width, im.height, im.frames, im.channels);

    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                // search outwards until we have sufficient alpha
                float alpha = 0;

                int r;
                for (r = 0; alpha < 1; r++) {

                    // find all dx, dy, dt at radius sqrt(r)
                    int sqrtR = (int)(sqrt((float)r));
                    int minDt = max(-sqrtR, -t), maxDt = min(sqrtR, im.frames-1-t);
                    int minDy = max(-sqrtR, -y), maxDy = min(sqrtR, im.height-1-y);
                    int minDx = max(-sqrtR, -x), maxDx = min(sqrtR, im.width-1-x);

                    for (int dt = minDt; dt <= maxDt; dt++) {
                        for (int dy = minDy; dy <= maxDy; dy++) {
                            for (int dx = minDx; dx <= maxDx; dx++) {
                                int R = dx * dx + dy * dy + dt * dt;
                                if (R == r) { 
                                    alpha += mask(x+dx, y+dy, t+dt)[0];
                                    for (int c = 0; c < im.channels; c++) {
                                        out(x, y, t)[c] += (im(x+dx, y+dy, t+dt)[c] * 
                                                                   mask(x+dx, y+dy, t+dt)[0]);
                                    }
                                } else if (R < r && dx < 0) dx *= -1; // optimization, skip over low values of |dx|
                            }
                        }
                    }


                }

                for (int c = 0; c < im.channels; c++) out(x, y, t)[c] /= alpha;
            }
        }
    }

    return out;
}




#include "footer.h"
