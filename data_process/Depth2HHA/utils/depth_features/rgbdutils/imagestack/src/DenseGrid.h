#ifndef IMAGESTACK_DENSE_GRID_H
#define IMAGESTACK_DENSE_GRID_H
#include "header.h"

/******************************************************************
 * This is the Bilateral Grid from the paper:                     *
 * A Fast Approximation of the Bilateral Filter using a Signal    *
 * Processing Approach                                            *
 * Sylvain Paris and Fredo Durand                                 *
 *                                                                *
 * The implementation is by Andrew Adams, and uses multilinear    *
 * splatting instead of nearest neighbour, which is slightly      *
 * slower but produces more accurate results.                     *
 ******************************************************************/

class DenseGrid {
  public:

    static Image filter(Image im, Image ref, float accuracy, size_t *memory) {
        int taps;
        if (accuracy < 0.25) {
            taps = 1;
            // variance = 0
        } else if (accuracy < 0.5) {
            taps = 3;
            // tent
            // [1 2 1]/4
            // variance = 0.5
        } else if (accuracy < 0.75) {
            taps = 5;
            // quadratic
            // [1 4 6 4 1]/16
            // variance = 1
        } else {
            taps = 7;
            // cubic
            // [1 6 15 20 15 6 1]/64
            // variance = 1.5
        }       

        DenseGrid grid(ref.channels, im.channels+1, taps);

        //printf("Allocating...\n");
        float *refPtr = ref(0, 0, 0);
        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {
                for (int x = 0; x < im.width; x++) {
                    grid.preview(refPtr);
                    refPtr += ref.channels;
                }
            }
        }
    
        //printf("Splatting...\n");

        float *col = new float[im.channels+1];
        col[im.channels] = 1;
        
        float *imPtr = im(0, 0, 0);
        refPtr = ref(0, 0, 0);
        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {
                for (int x = 0; x < im.width; x++) {
                    for (int c = 0; c < im.channels; c++) {
                        col[c] = *imPtr++;
                    }
                    grid.splat(refPtr, col);
                    refPtr += ref.channels;
                }
            }
        }
    
        //printf("Blurring...\n");
    
        grid.blur();
    
        //printf("Slicing...\n");  
        Image out = im.copy();
        
        float *outPtr = out(0, 0, 0);
        refPtr = ref(0, 0, 0);
        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {
                for (int x = 0; x < im.width; x++) {
                    grid.slice(refPtr, col);
                    float scale = 1.0f/col[im.channels];
                    for (int c = 0; c < im.channels; c++) {
                        *outPtr++ = col[c]*scale;
                    }
                    refPtr += ref.channels;
                }
            }
        }

        *memory = grid.memoryUsed();
        
        return out;
    }
    
    DenseGrid(int d_, int vd_, int taps_ = 3) : d(d_), vd(vd_), taps(taps_) {
        scaleFactor = new float[d];
        positionF = new float[d];
        positionFInv = new float[d];
        positionI = new int[d];
        minPosition = NULL;
        maxPosition = NULL;
        stride = NULL;
        grid = NULL;

        for (int i = 0; i < d; i++) {
            // The kernel for a single multi-linear interpolation is a
            // cube convolved with itself. Therefore it's variance is
            // twice the total variance of a d-dimensional unit cube.

            // total variance of a cube = d/12
            // total variance of splatting = d/6
            // total variance of splatting + slicing = d/3

            // total variance of the blur step is d(taps-1)/4
            
            // so scale factor should be the std dev in each dimension
            // = sqrt(total variance / d) = sqrt(1/3 + (taps-1)/4)
            
            scaleFactor[i] = sqrtf(1.0/3 + (taps-1)*0.25); 
        }

    }

    ~DenseGrid() {
        delete[] scaleFactor;
        delete[] positionF;
        delete[] positionFInv;
        delete[] positionI;
        delete[] minPosition;
        delete[] maxPosition;
        delete[] stride;
        delete[] grid;
    }

    void preview(float *position) {
        if (!minPosition) {
            minPosition = new float[d];
            maxPosition = new float[d];
            for (int i = 0; i < d; i++) {
                minPosition[i] = position[i]*scaleFactor[i];
                maxPosition[i] = position[i]*scaleFactor[i];
            }
        } else {
            for (int i = 0; i < d; i++) {
                if (position[i]*scaleFactor[i] < minPosition[i])
                    minPosition[i] = position[i]*scaleFactor[i];
                if (position[i]*scaleFactor[i] > maxPosition[i])
                    maxPosition[i] = position[i]*scaleFactor[i];
            }
        }        
    }
    
    void splat(float *position, float *value) {
        if (!grid) {
            stride = new int[d+1];
            sizes = new int[d];
            stride[0] = vd;
            for (int i = 0; i < d; i++) {
                sizes[i] = (int)(ceil(maxPosition[i] - minPosition[i])+1);
                stride[i+1] = stride[i]*sizes[i];
            }            
            grid = new float[stride[d]];
            memset(grid, 0, sizeof(float)*stride[d]);

        }

        query<true>(position, value);
    }

    void blur() {
        switch(taps) {
        case 3:
            blur_<3>();
            return;
        case 5:
            blur_<5>();
            return;
        case 7:
            blur_<7>();
            return;
        }
    }
    
    void slice(float *position, float *value) {
        query<false>(position, value);
    }

    size_t memoryUsed() {
        return (sizeof(float)*stride[d]);
    }

  private:

    template <int taps_>
    void blur_() {
        int *rowLocation = new int[d];
        float *tmp1 = new float[vd];
        float *tmp2 = new float[vd];
        
        for (int j = 0; j < d; j++) {            

            for (int i = 0; i < d; i++) rowLocation[i] = 0;

            //printf("Blurring in direction %d\n", j);
            // iterate through every dimension except for j and the value dimension            
            for (int row = 0; row < stride[d]/(vd*sizes[j]); row++) {
                
                float *startOfRow = grid;
                for (int i = 0; i < d; i++) {
                    startOfRow += rowLocation[i]*stride[i];
                }

                for (int iters = 0; iters < taps_/2; iters++) {
                    float *rowPtr = startOfRow;

                    // now walk along the row blurring
                    int s = stride[j];
                    
                    for (int k = 0; k < vd; k++) {
                        tmp1[k] = rowPtr[k]/2;
                    }
                    
                    for (int i = 0; i < sizes[j]-1; i++) {
                        // the average of previous + current is stored in tmp1
                        
                        // save the average of current + next into tmp2
                        for (int k = 0; k < vd; k++) {
                            tmp2[k] = 0.5*(rowPtr[k] + rowPtr[k+s]);
                        }
                        
                        // then clobber current with the average of tmp1 and tmp2
                        for (int k = 0; k < vd; k++) {
                            rowPtr[k] = 0.5*(tmp1[k] + tmp2[k]);
                        }
                        
                        // switch tmp1 and tmp2
                        float *tmp3 = tmp1;
                        tmp1 = tmp2;
                        tmp2 = tmp3;
                        
                        rowPtr += s;
                    }
                    
                    // write the last one
                    for (int k = 0; k < vd; k++) {
                        rowPtr[k] = 0.5*(tmp1[k] + 0.5*rowPtr[k]);
                    }
                }

                // This is basically addition with ripple carry,
                // ignoring the digit in the place of the current
                // dimension (digit j)
                int k = 0;
                if (k == j) k++;
                if (k >= d) printf("PANIC!\n");
                rowLocation[k]++;

                while (rowLocation[k] == sizes[k]) {
                    rowLocation[k] = 0;
                    k++;
                    if (k == j) k++;
                    if (k >= d) break;
                    rowLocation[k]++;
                }               

            }
        }
        delete rowLocation;
        delete tmp1;
        delete tmp2;
    }

    template<bool splatting>
    void query(float *position, float *value) {

        // break the query into integral and floating point portions
        for (int i = 0; i < d; i++) {
            float f = (position[i]*scaleFactor[i] - minPosition[i]);
            positionI[i] = (short)floorf(f);
            positionF[i] = f - positionI[i];
            positionFInv[i] = 1 - positionF[i];
        }

        if (!splatting) {
            for (int j = 0; j < vd; j++) {
                value[j] = 0;
            }
        }

        // find the position of the top left
        float *topLeft = grid;
        for (int i = 0; i < d; i++) {
            topLeft += positionI[i]*stride[i];
        }

        // iterate through the neighbours
        for (int i = 0; i < (1<<d); i++) {
            float weight = 1;
            float *val = topLeft;
            for (int j = 0; j < d; j++) {
                if (i & (1 << j)) {
                    val += stride[j];
                    weight *= positionF[j];
                } else {
                    weight *= positionFInv[j];
                }
            }

            if (splatting) {
                for (int j = 0; j < vd; j++) {
                    val[j] += weight*value[j];
                }
            } else if (val) {
                for (int j = 0; j < vd; j++) {
                    value[j] += weight*val[j];                
                }
            }
        }
    }

  private:

    int d, vd, taps;
    float *scaleFactor;
    float *positionF, *positionFInv;
    float *grid;
    float *minPosition, *maxPosition;
    int *positionI;
    int *stride;
    int *sizes;
};

#include "footer.h"
#endif


