#include "main.h"
#include "HDR.h"
#include "Arithmetic.h"
#include "File.h"
#include <fstream>
#include "header.h"

//#define HDR_DEBUG

void AssembleHDR::help() {
    printf("-assemblehdr takes a volume in which each frame is a linear luminance image\n"
           "taken at a different exposure, and compiles them all into a single HDR image,\n"
           "gracefully handling oversaturated regions.\n\n"
           "If exposure values are known, they can be given in increasing frame order.\n"
           "Otherwise, assemblehdr attempts to discover the exposure ratios itself, which\n"
           "may fail if there are very few pixels that are properly imaged in multiple\n"
           "frames. For best results, load the frames in either increasing or decreasing\n"
           "exposure order.\n\n"
           "Usage: ImageStack -loadframes input*.jpg -gamma 0.45 -assemblehdr -save out.exr\n"
           "   or  ImageStack -loadframes input*.jpg -gamma 0.45 -assemblehdr 1.0 0.5 0.1\n"
           "                  -save output.exr\n\n");
    
}

void AssembleHDR::parse(vector<string> args) {
  
    assert(args.size() == 0 ||
           args.size() >= static_cast<unsigned int>(stack(0).frames), 
           "-assemblehdr takes zero arguments or an exposure value for each frame in the volume (plus an optional gamma adjustment) \n");    
    if (args.size() == 0) {
        Image im = apply(stack(0));
        pop();
        push(im);
    }
    else {
        vector<float> exposures(stack(0).frames);
        string gamma = "1.0";
        if (static_cast<unsigned int>(stack(0).frames) < args.size()) {
            gamma = args[args.size() - 1];
        }
        for (unsigned int e = 0; e < static_cast<unsigned int>(stack(0).frames); e++) {
            exposures[e]=readFloat(args[e]);
        }
        Image im = apply(stack(0), exposures, gamma);
        pop();
        push(im);
    }
}


struct gammaInfo {
    enum gammaType { NONE, FLOAT, MAP };
    gammaType type;
    float gamma;
    float R[256];
    float G[256];
    float B[256];
};

Image AssembleHDR::apply(Window frames, vector<float> &exposures, string gamma) {    
    assert(frames.frames == (int)exposures.size(), "AssembleHDR::applyKnownExposures - mismatched exposure and frame counts");
    // Figure out gamma conversion
    gammaInfo gi;
    gi.gamma = 1.0; // to get rid of warning
    try {
        // Try to convert parameter to float
        gi.gamma = readFloat(gamma);
        if (gi.gamma == 1.0f) {
            gi.type = gammaInfo::NONE;
        } else {
            gi.type = gammaInfo::FLOAT;
            printf("Using gamma of %f to reverse the camera response curve\n", gi.gamma);
        }
    } catch(Exception) {
        // Exposure isn't a float - assume it's a file name for a camera curve as output by HDRShop
        ::std::ifstream curveFile(gamma.c_str());
        assert(curveFile.is_open(), ("Can't open filename "+gamma+" for reading a camera curve").c_str());

        gi.type = gammaInfo::MAP;

        // Parse file - first line is C = [
        // Then there are 256 lines of float triplets in RGB order
        // and the final line is ];
      
        curveFile.ignore(100,'\n');

        for ( unsigned int m =0; m < 256; m++) {
            float rr, gg, bb;
            curveFile >> rr >> gg >> bb;
            // curve.m data is in log space, and needs normalization too
            gi.R[m] = expf(rr)/4.0; // 4.0 mapping arbitrarily determined from HDRShop output view
            gi.G[m] = expf(gg)/4.0;
            gi.B[m] = expf(bb)/4.0;
            //        printf("%f %f %f\n", gi.R[m], gi.G[m], gi.B[m]);
            //        fflush(stdout);

        }            
        printf("Using file %s for camera curve\n", gamma.c_str());
        curveFile.close();
    }


    Image out(frames.width, frames.height, 1, frames.channels);
    Image weight(frames.width, frames.height, 1, 1);
  
    float maxExpVal = exposures[0];
    float minExpVal = exposures[0];
    for (unsigned int e = 1; e < exposures.size(); e++) {
        if ( exposures[e] > maxExpVal ) maxExpVal = exposures[e];
        if ( exposures[e] < minExpVal ) minExpVal = exposures[e];
    }


    // Debug code to print out a pixel value and weights
#ifdef HDR_DEBUG
    int x_debug=1640, y_debug=1045;
#endif

    for (int t = 0; t < frames.frames; t++) {    
        float exposureRatio = 1/exposures[t];

        printf("Frame %d: Exposure %f", t, exposures[t]);
        cutoffType cutoff; 
        if ( fabs(exposures[t] -maxExpVal) < 0.001 * maxExpVal) { 
            printf("  - Longest exposure");
            cutoff = LONGEST_EXPOSURE;
        } else if ( fabs(exposures[t] - minExpVal) < 0.001 * minExpVal ) {
            printf("  - Shortest exposure");
            cutoff = SHORTEST_EXPOSURE;
        } else {
            cutoff = REGULAR;
        }
        int count=0;
        for (int y = 0; y < frames.height; y++) {      
            for (int x = 0; x < frames.width; x++) {
                float w = weightFunc(frames(x, y, t), frames.channels, cutoff);
                if (w < 1.0) count++;
                weight(x,y)[0] += w;
                if (gi.type == gammaInfo::NONE) {
                    for (int c = 0; c < frames.channels; c++) {
                        out(x,y)[c] += w * exposureRatio * frames(x, y, t)[c];
                    }
                } else if (gi.type == gammaInfo::FLOAT) {
                    for (int c = 0; c < frames.channels; c++) {
                        out(x,y)[c] += w * exposureRatio * powf(frames(x, y, t)[c], gi.gamma);
                    }
                } else if (gi.type == gammaInfo::MAP) {
                    out(x,y)[0] += w * exposureRatio * gi.R[HDRtoLDR(frames(x,y,t)[0])];
                    out(x,y)[1] += w * exposureRatio * gi.G[HDRtoLDR(frames(x,y,t)[1])];
                    out(x,y)[2] += w * exposureRatio * gi.B[HDRtoLDR(frames(x,y,t)[2])];
                }

            }
        }
        printf(".  %d of %d pixels weighted 1.0 (%f%%)\n", 
               frames.height*frames.width-count,
               frames.height*frames.width, 
               100*((float)(frames.height*frames.width-count)/(frames.height*frames.width)));
    }

    Divide::apply(out, weight);

    return out;
}

Image AssembleHDR::apply(Window frames) {

    vector<float> ratios(frames.frames-1);

    Image out(frames.width, frames.height, 1, frames.channels);
    Image weight(frames.width, frames.height, 1, 1);

    // Find max and min exposure frames

    int maxExpFrame = 0;
    double minMean = 0;    
    int minExpFrame = 0;
    double maxMean = 0;

    for (int t = 0; t < frames.frames; t++) {
        double mean = 0;
        for (int y = 0; y < frames.height; y++) {
              for (int x = 0; x < frames.width; x++) {
                mean += static_cast<double>(frames(x, y, t)[0]);
            }
        }
        if (t == 0) {
            minMean = mean;
            maxMean = mean;
        } else {
            if (mean < minMean) {
                minExpFrame = t;
                minMean = mean;
            } 
            if (mean > maxMean) {
                maxExpFrame = t;
                maxMean = mean;
            }
        }
    }

    // initialize the output to the first frame
    cutoffType cutoff;
    printf("Frame 0 scale is 1.0");
    if (0 == maxExpFrame) { 
        printf(" - Longest exposure\n");
        cutoff = LONGEST_EXPOSURE;
    } else if ( 0 == minExpFrame) {
        printf(" - Shortest exposure\n");
        cutoff = SHORTEST_EXPOSURE;
    } else {
        printf("\n");
        cutoff = REGULAR;
    }

    for (int y = 0; y < frames.height; y++) {
        for (int x = 0; x < frames.width; x++) {
            float w = weightFunc(frames(x, y, 0), frames.channels, cutoff);
            weight(x, y)[0] = w;
            for (int c = 0; c < frames.channels; c++) {
                out(x, y)[c] = w*frames(x, y, 0)[c];
            }
        }
    }
           
    for (int t = 0; t < frames.frames - 1; t++) {

        // Check for special cutoffs
        if (t+1 == maxExpFrame) { 
            cutoff = LONGEST_EXPOSURE;
        } else if ( t+1 == minExpFrame) {
            cutoff = SHORTEST_EXPOSURE;
        } else {
            cutoff = REGULAR;
        }
    
        // find the average ratio between each frame and the current output
        float count = 0;
        float ratio = 0;
        for (int y = 0; y < frames.height; y++) {
            for (int x = 0; x < frames.width; x++) {
                float weightOld = weight(x, y)[0];
                if (weightOld < 1.0) 
                    continue;
                if (weightFunc(frames(x, y, t+1), frames.channels, cutoff) < 1.0 ) 
                    continue;
                for (int c = 0; c < frames.channels; c++) {
                    float valOld = out(x, y)[c] / weightOld;
                    float valNew = frames(x, y, t+1)[c];
                    if (valNew < 0.1 || valOld < 0.1) continue; // Some color channels may be zero even for good pixels
                    ratio += valOld / valNew;
                    count++;
                }
            }
        }
        ratio /= count;

        printf("Frame %i scale is %f", t+1, ratio);
        switch(cutoff) {
        case LONGEST_EXPOSURE:
            printf(" - Longest exposure\n");
            break;
        case SHORTEST_EXPOSURE:
            printf(" - Shortest exposure\n");
            break;
        case REGULAR:
            printf("\n");
            break;
        }
        // add that frame in
        for (int y = 0; y < frames.height; y++) {
            for (int x = 0; x < frames.width; x++) {
                float w = weightFunc(frames(x, y, t+1), frames.channels, cutoff);
                weight(x, y)[0] += w;
                for (int c = 0; c < frames.channels; c++) {
                    out(x, y)[c] += w * ratio * frames(x, y, t+1)[c];
                }
            }
        }
    }

    Divide::apply(out, weight);
    return out;

}

float AssembleHDR::weightFunc(float *x, int channels, cutoffType cutoff) {
    
    // Weighting function rationale: A pixel is well-captured if its _highest_ 
    // color channel value is in the linear range of the sensor (varies by the 
    // sensor, of course, but 0.1-0.9 is a likely good range)
    // However, if we are dealing with the longest exposure in the stack of images,
    // there can be no better value for underexposured pixels - so keep weights at 1
    // all the way to 0.  Similarly, if the image is the one with the shortest exposure,
    // the oversaturated values it contains are still the best estimates that can be found
    // in the stack.  So don't roll off the high end values
    // And finally, to eliminate truly gone pixels from the weighting, drop the weight
    // to zero just a bit before the extremes.  This avoids problems where a small
    // weight is divided by a very short exposure to result in a large value and noise.

    const float lowCutoffStart = 0.15;   // Was 0.2, modified on 08/28/2008 // Was 0.1, modified on 12/14/2006
    const float lowCutoffEnd = 0.05;   // Was 0.1, modified on 08/28/2008 // Was 0.001, modified on 12/14/2006
    const float highCutoffStart = 0.85;  // Was 0.8, modified on 08/28/2008 // Was 0.9, modified on 12/14/2006
    const float highCutoffEnd = 0.95;   // Was 0.9. modified on 08/28/2008 // Was 0.999, modified on 12/14/2006

    // Precalculate to speed things up
    const float lowCutoffScale = 1/(lowCutoffStart-lowCutoffEnd);
    const float highCutoffScale = 1/(highCutoffEnd-highCutoffStart);

    // Find maximum channel value
    float maxVal = x[0];
    for (int c = 1; c < channels; c++) {
        if (x[c] > maxVal) maxVal = x[c];
    }

    // Calculate weight - trapezoidal shape that falls to zero before 0 or 1
    if (cutoff != LONGEST_EXPOSURE) {
        if (maxVal <= lowCutoffEnd) 
            return 0.0;
        else if (maxVal < lowCutoffStart)
            return (maxVal-lowCutoffEnd)*lowCutoffScale;
    } 
    if (cutoff != SHORTEST_EXPOSURE) {
        if (maxVal >= highCutoffEnd)
            return 0.0;
        else if (maxVal > highCutoffStart)
            return 1.0-(maxVal-highCutoffStart)*highCutoffScale;
    } 
    return 1.0;
}
#include "footer.h"
