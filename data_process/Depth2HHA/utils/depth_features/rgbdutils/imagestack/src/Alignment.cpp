#include "main.h"
#include "Alignment.h"
#include "Statistics.h"
#include "Filter.h"
#include "Arithmetic.h"
#include "File.h"
#include "Color.h"
#include "LinearAlgebra.h"
#include <algorithm>
#include "Geometry.h"
#include "Convolve.h"
#include "Display.h"
#include "header.h"

// First we define the various types of transformations we may wish to
// solve for. All of these classes lean heavily on the least squares
// solving in LinearAlgebra.h
class Transform {
public:
    virtual ~Transform() {}
    
    // Add a new constraint (ie (x1,y1) must warp to (x2, y2))
    virtual void addCorrespondence(float x1, float y1, float x2, float y2) = 0;

    // Once all the constraints are added, solve for the optimal warp
    virtual void solve() = 0;

    // After solving, we can apply the warp to a given (x1, y1) to produce (x2, y2)
    virtual void apply(float x1, float y1, float *x2, float *y2) = 0;

    // Different types of transformations required different numbers
    // of constraints in order to produce a model
    virtual int constraintsRequired() = 0;

    // Adjust scale factors if transform is calculated with downsampled input imagess
    virtual void adjustDownsampleScale(int a1, int a2) = 0;

    // Forget all the correspondences so far, and start again. This is
    // useful in RANSAC if we picked bad initial correspondences.
    virtual void reset() = 0;
};

// Solving for a least squares translation is easy, we just average
// the translations of all the correspondences given.
class Translation : public Transform {
public:
    Translation() {
        reset();
    }

    virtual ~Translation() {}

    void reset() {
        dx = dy = dxSum = dySum = 0;
        count = 0;
    }

    void addCorrespondence(float x1, float y1, float x2, float y2) {
        dxSum += x2 - x1;
        dySum += y2 - y1;
        count++;
    }

    void solve() {
        dx = dxSum / count;
        dy = dySum / count;
    }

    int constraintsRequired() {
        return 1;
    }

    void apply(float x1, float y1, float *x2, float *y2) {
        *x2 = x1 + dx;
        *y2 = y1 + dy;
    }

    void adjustDownsampleScale(int s1, int s2) {
        dx /= s1;
        dy /= s1;
        dx *= s2;
        dy *= s2;
    }

private:
    // the transform parameters
    float dx, dy;
    // the internal state to keep track of
    float dxSum, dySum;
    int count;
};

// Solving for a least squares similarity (ie rotation, translation,
// and scale) matrix is a little tricker. There are four free
// parameters. We use the LeastSquares solver class to solve it.
class Similarity : public Transform {
public:
    Similarity() {
        reset();
    }

    virtual ~Similarity() {}

    void reset() {
        params[0] = params[1] = params[2] = params[3] = 0;
        solver.reset();
    }

    void addCorrespondence(float x1, float y1, float x2, float y2) {
        {
            float in[4] = {x1, y1, 1, 0};
            float out[1] = {x2};
            solver.addCorrespondence(in, out);
        }
        {
            float in[4] = {y1, -x1, 0, 1};
            float out[1] = {y2};
            solver.addCorrespondence(in, out);
        }
    }

    void solve() {
        solver.solve(params);
    }

    int constraintsRequired() {
        return 2;
    }

    void apply(float x1, float y1, float *x2, float *y2) {
        *x2 = params[0]*x1 + params[1]*y1 + params[2];
        *y2 = params[0]*y1 - params[1]*x1 + params[3];
    }

    void adjustDownsampleScale(int s1, int s2){
        params[2] /= s1;
        params[3] /= s1;
        params[2] *= s2;
        params[3] *= s2;
    }

protected:
    // the transform parameters
    double params[4];

    // the internal state
    LeastSquaresSolver<4, 1> solver;        
};

// Solving for a 2D rigid transformation (rotation and translation but
// no scale) is actually surprisingly hard. We dodge the problem by
// just solving for a similarity transform and altering the parameters
// to remove the scale factor.
class Rigid : public Similarity {
public:
    virtual ~Rigid() {}

    void solve() {
        solver.solve(params);
        double l = ::sqrt(params[0]*params[0] + params[1]*params[1]);
        params[0] /= l;
        params[1] /= l;
    }
};

// Solving for an affine transform is a fairly standard least squares
// solve, because it's easy to represent an affine transform as a
// 2x3 matrix.
class Affine : public Transform {
public:
    Affine() {
        reset();
    }

    virtual ~Affine() {}

    void reset() {
        params[0] = params[1] = params[2] = params[3] = params[4] = params[5] = 0;
        solver.reset();
    }

    void addCorrespondence(float x1, float y1, float x2, float y2) {
        float in[3] = {x1, y1, 1};
        float out[2] = {x2, y2};
        solver.addCorrespondence(in, out);
    }

    void solve() {
        solver.solve(params);
    }

    int constraintsRequired() {
        return 3;
    }

    void apply(float x1, float y1, float *x2, float *y2) {
        *x2 = params[0]*x1 + params[2]*y1 + params[4];
        *y2 = params[1]*x1 + params[3]*y1 + params[5];
    }

    void adjustDownsampleScale(int s1, int s2) {
        params[4] /= s1;
        params[5] /= s1;
        params[4] *= s2;
        params[5] *= s2;
    }

private:
    // the transform parameters
    double params[6];

    // the internal state
    LeastSquaresSolver<3, 2> solver;        
};

// Solving for 2D perspective warps involves some algebraic
// manipulation.  In general, a 2D perspective warp can be expressed
// by a 3x3 matrix, which maps (x1, y1, 1) to some homogeneous
// representation of (x2, y2) - ie (w.x2, w.y2, w).
//
// If you write this out and substitute out w, then shuffle terms
// around to make it linear, you can solve for the eight parameters
// of the transform. Why only 8 parameters in a 3x3 matrix? Because
// its outputs are homogeneous vectors, the matrix is invariant to
// scale, so we can assume WLOG that the bottom right entry is 1.

class Perspective : public Transform {
public:
    Perspective() {
        reset();
    }

    virtual ~Perspective() {}
 
    void reset() {
        for (int i = 0; i < 8; i++) params[i] = 0;
        solver.reset();
    }

    void addCorrespondence(float x1, float y1, float x2, float y2) {
        x1 /= 1000;
        y1 /= 1000;
        x2 /= 1000;
        y2 /= 1000;
        {
            float in[8] = {-x1*x2, -y1*x2, x1, y1, 1, 0, 0, 0};
            float out[1] = {x2};
            solver.addCorrespondence(in, out);
        }
        {
            float in[8] = {-x1*y2, -y1*y2, 0, 0, 0, x1, y1, 1};
            float out[1] = {y2};
            solver.addCorrespondence(in, out);
        }
    }

    void solve() {
        solver.solve(params);
    }

    int constraintsRequired() {
        return 4;
    }

    void apply(float x1, float y1, float *x2, float *y2) {
        x1 /= 1000;
        y1 /= 1000;        
        float w = 1.0f/(params[0]*x1 + params[1]*y1 + 1);        
        *x2 = 1000*(params[2]*x1 + params[3]*y1 + params[4])*w;
        *y2 = 1000*(params[5]*x1 + params[6]*y1 + params[7])*w;
    }

    void getParams(double* p) {
        for(int i=0;i<8;i++) {
            p[i] = params[i];
        }
    }
    
    void setParams(double* p) {
        double *Ptr = p;
        for(int i=0;i<8;i++) {
            params[i] = *Ptr++;
        }
    }

    void adjustDownsampleScale(int s1, int s2){

        params[0]/=s1; params[1]/=s1; params[2]/=s1;
        params[3]/=s1; params[5]/=s1; params[6]/=s1;
        params[2]*=s2; params[3]*=s2; params[4]*=s2;
        params[5]*=s2; params[6]*=s2; params[7]*=s2;

        /*
        if(a2!=0) {
            float c2 = 0.5*a2-0.5;
            params[2]*=a2; params[3]*=a2; params[4]*=a2;
            params[5]*=a2; params[6]*=a2; params[7]*=a2;
            params[2]+=c2*params[0]; params[5]+=c2*params[0];
            params[3]+=c2*params[1]; params[6]+=c2*params[1];
            params[4]+=c2; params[7]+=c2;
        }

        if(a1!=0) {
            float w;
            float c1 = 1/(2*a1)-0.5;
            params[0]/=a1; params[1]/=a1; params[2]/=a1;
            params[3]/=a1; params[5]/=a1; params[6]/=a1;
            params[4]+=c1*(params[2]+params[3]);
            params[7]+=c1*(params[5]+params[6]);
            w = 1/(c1*(params[0]+params[1])+1);
            for(int i=0;i<8;i++){
                params[i]*=w;
            }
        }
        */
    }

private:
    // the transform parameters
    double params[8];

    // the internal state
    LeastSquaresSolver<8, 1> solver;            
};

// A Digest is a data structure that gathers together all the features
// extracted from a single image.
class Digest {
public:

    void findOrientations(LocalMaxima::Maximum m, 
                          Image* magPyramid, Image* ornPyramid, 
                          vector<float> *sigma, vector<float> *orientations) {

        //printf("Start finding orientations..\n");

        int it = (int)(m.t + 0.5);
        if (it < 1) it = 1;

        // Find major orientation in 16x16 patch
        // Each sample is weighted by its gradient magnitude and falloff mask
        float hist[36];
        for(int i=0;i<36;i++) {
            hist[i] = 0;
        }
        for(int i=0;i<16;i++) {
            for(int j=0;j<16;j++) {
                float gridX = m.x + i - 7.5;
                float gridY = m.y + j - 7.5;
                float weight = fastexp(-((i-7.5)*(i-7.5)+(j-7.5)*(j-7.5)) / ( 2*(1.5*(*sigma)[it+1])*(1.5*(*sigma)[it+1]) ));
                float value;
                ornPyramid[it-1].sample2DLinear(gridX, gridY, &value);
                int index = floor(( value + M_PI) * 36 / (2 * M_PI));
                magPyramid[it-1].sample2DLinear(gridX, gridY, &value);
                hist[index] += value * weight;
            }
        }

        // Find local maxima of histogram with local refinement
        vector<LocalMaxima::Maximum> maxima;
        LocalMaxima::Maximum h;
        if(hist[0]>hist[1] && hist[0]>hist[35]) {
            h.value = hist[0];
            h.x = (hist[1]-hist[35])/(hist[35]+hist[1]+hist[0]);
            maxima.push_back(h);
        }
        for(int i=1;i<35;i++) {
            if(hist[i]>hist[i-1] && hist[i]>hist[i+1]) {
                h.value = hist[i];
                h.x = i + (hist[i+1]-hist[i-1])/(hist[i-1]+hist[i+1]+hist[i]);
                maxima.push_back(h);
            }
        }
        if(hist[35]>hist[34] && hist[35]>hist[0]) {
            h.value = hist[35];
            h.x = 35 + (hist[0]-hist[34])/(hist[34]+hist[0]+hist[35]);
            maxima.push_back(h);
        }
        ::std::sort(maxima.begin(), maxima.end());
        

        // Pick major orientation
        // Include all local maxima that are over 80% of the maximum peak
        int idx = maxima.size()-1;
        float max_hist = maxima[idx].value;
        while(maxima[idx].value >= 0.8 * max_hist) {
            float orientation = maxima[idx].x/36*2*M_PI - M_PI;
            (*orientations).push_back(orientation);
            idx--;
            if(idx<0) {break;}
        }

        //printf("Orientation asigned..\n");
    }


    // 128-bit SIFT-like descriptor.
    struct Descriptor {
    public:
        Descriptor() {}
        
        Descriptor(LocalMaxima::Maximum m,
                   Image* magPyramid, Image* ornPyramid,
                   vector<float> *sigma, float orientation) {
    
            length = 128;

            int it = (int)(m.t + 0.5);
            if (it < 1) it = 1;

            // Find 128-sift-like descriptor in 16x16 patch
            // Rotate sampling grid by major orientation to achieve rotation invariance
            // Each sample is weighted by its gradient magnitude and falloff mask
            float *dPtr = &(desc[0]);
            for(int i=0;i<4;i++) {
                for(int j=0;j<4;j++) {
                    
                    float hist[8];
                    for(int h=0;h<8;h++) {hist[h]=0;}

                    for(int k=0;k<4;k++) {
                        for(int l=0;l<4;l++) {

                            float gridX = (4*j-6) + (l-1.5);
                            float gridY = (4*i-6) + (k-1.5);
                            
                            float rotX = cos(orientation)*gridX  - sin(orientation)*gridY;
                            float rotY = sin(orientation)*gridX  + cos(orientation)*gridY;

                            float weight = fastexp(-(gridX*gridX+gridY*gridY) / ( 2*(1.5*(*sigma)[it+1])*(1.5*(*sigma)[it+1]) ));
                            float value;
                            ornPyramid[it-1].sample2DLinear(m.x+rotX, m.y+rotY, &value);
                            value -= orientation;
                            value = value<-M_PI ? value+2*M_PI : value>M_PI ? value-2*M_PI : value;
                            
                            int ivalue = floor((value+M_PI) * 8 / (2 * M_PI));
                            magPyramid[it-1].sample2DLinear(m.x+rotX, m.y+rotY, &value);
                            
                            hist[ivalue] += value * weight;
                            
                        }
                    }
                    
                    for(int h=0;h<8;h++) {
                        *dPtr++ = hist[h];
                    }
                }
            }

            // Normalize descriptor vector to have a unit magnitude
            // Limit maximum component to 0.2 in the first stage and normalize again
            // to favor orientation distribution rather than peak matching
            float norm=0;
            for(int i=0;i<128;i++) {
                norm += desc[i]*desc[i];
            }
            norm = sqrt(norm);
            for(int i=0;i<128;i++) {
                desc[i] /= norm;
                desc[i] = desc[i]>0.2 ? 0.2 : desc[i];
            }
            norm=0;
            for(int i=0;i<128;i++) {
                norm += desc[i]*desc[i];
            }
            norm = sqrt(norm);
            for(int i=0;i<128;i++) {
                desc[i] /= norm;
            }

            //printf("Descriptor assigned..\n");
        }

        int length;
        float desc[128];
    };

    struct Feature : public LocalMaxima::Maximum {
    public:
        Feature(LocalMaxima::Maximum m, Image* magPyramid, Image* ornPyramid, vector<float> *sigma, float orientation) {
            x = m.x;
            y = m.y;
            t = floor(m.t + 0.5);
            
            value = m.value;
            descriptor = Descriptor(m, magPyramid, ornPyramid, sigma, orientation);
            usage = 0;
        }
        
        // Distance between two features is the sum of squared differences between the two descriptors
        float distance(Feature *other) {
            float dist = 0;
            
            float *thisPtr = &(descriptor.desc[0]);
            float *otherPtr = &(other->descriptor.desc[0]);
            for(int i = 0; i < descriptor.length; i++){
                float d = *thisPtr++ - *otherPtr++;
                dist += d*d;
            }
            return dist;
        }
        
        // It's useful to keep track of how many times any one given
        // features is used, so we don't depend too heavily on a
        // single feature.
        int usage;        

        bool usePatch;
        Window patch;
        Descriptor descriptor;
    };

    // A correspondences is a pair of features that hopefully match.
    struct Correspondence {
        Correspondence(Feature *a_, Feature *b_) {
            a = a_;
            b = b_;
            distance = a->distance(b);
        }
        float distance;
        Feature *a, *b;

        // Correspondences with lower distances between their features
        // are better, so there is an ordering on correspondences.
        bool operator<(const Correspondence &other) const {
            return distance < other.distance;
        }
    };
    
    Digest(Window im) {

        // Convert to grayscale
        vector<float> grayMatrix;
        for (int i = 0; i < im.channels; i++) {
            grayMatrix.push_back(1.0f/im.channels);
        }
         Image gray = ColorMatrix::apply(im, grayMatrix);

        // Gaussian Pyramid
        // k1: first sigma, k: scale factor between each level
        // GAUSSIAN_LEVELS: number of pyramid levels
        float k1 = 1.6f, k = sqrtf(2.0f);

        const int gaussianLevels = 5;

        vector<float> sigma;
        Image magPyramid[gaussianLevels-3];
        Image ornPyramid[gaussianLevels-3];
        Image gPyramid = Upsample::apply(gray, 1, 1, gaussianLevels);

        float sig = k1;
        for(int i = 0; i < gaussianLevels; i++) {
            sigma.push_back(sig);
            Window level(gPyramid, 0, 0, i, gPyramid.width, gPyramid.height, 1);
            FastBlur::apply(level, sig, sig, 0);
            sig *= k;
        }

        // Magnitude and phase of gradient images
        for(int i=0;i<gaussianLevels-3;i++) {
            ornPyramid[i] = Image(gray.width, gray.height, gray.frames, gray.channels);
            magPyramid[i] = Image(gray.width, gray.height, gray.frames, gray.channels);

            for(int y=1;y<gray.height-1;y++) {
                
                Window level(gPyramid, 0, 0, i+2, gPyramid.width, gPyramid.height, 1);
                float *mPtr = magPyramid[i](1,y);
                float *oPtr = ornPyramid[i](1,y);
                float *dx1Ptr = level(0,y);
                float *dx2Ptr = level(2,y);
                float *dy1Ptr = level(1,y-1);
                float *dy2Ptr = level(1,y+1);
                
                for(int x=1;x<gray.width-1;x++) {
                    float dx = *dx1Ptr++ - *dx2Ptr++;
                    float dy = *dy1Ptr++ - *dy2Ptr++;
                    *mPtr++ = sqrt(dx*dx + dy*dy);
                    *oPtr++ = atan2f(dy, dx);
                }
            }
        }

        // DoG Pyramid
        for(int i = 0; i < gaussianLevels-1; i++) {
            Window thisLevel(gPyramid, 0, 0, i, gPyramid.width, gPyramid.height, 1);
            Window nextLevel(gPyramid, 0, 0, i+1, gPyramid.width, gPyramid.height, 1);
            Subtract::apply(thisLevel, nextLevel);
        }
        Window dogPyramid(gPyramid, 0, 0, 0, gPyramid.width, gPyramid.height, gaussianLevels-1);


        // Find local maxima
        vector<LocalMaxima::Maximum> maxima = LocalMaxima::apply(dogPyramid, true, true, true, 0.00001, 10);
        ::std::sort(maxima.begin(), maxima.end());

        //printf("Assigning features.. %d\n",maxima.size());

        for (int i = (int)maxima.size()-1, j=0; i >= 0 && j < 1024; i--, j++) {

            // Reject maxima that are too close to image boundary
            if (maxima[i].x < 20 || maxima[i].x > im.width-21 ||
                maxima[i].y < 20 || maxima[i].y > im.height-21) {
                j--;
                continue;
            }

            // Reject maxima that are located along edges
            int mt = (int)(maxima[i].t + 0.5);
            float mx = maxima[i].x; 
            float my = maxima[i].y;
            Image patch(3,3,1,1);
            float *ptr = patch(0,0,0);
            if(mt < 0 || mt >= gaussianLevels) {
                j--;
                continue;
            }

            for(int y=-1;y<=1;y++) {
                for(int x=-1;x<=1;x++) {
                    dogPyramid.sample2DLinear(mx+x, my+y, mt, ptr++);
                }
            }

            float Dxx = patch(0,1)[0] - 2 * patch(1,1)[0] + patch(2,1)[0];
            float Dyy = patch(1,0)[0] - 2 * patch(1,1)[0] + patch(1,2)[0];
            float Dxy = (patch(0,0)[0] - patch(0,2)[0] + patch(2,2)[0] - patch(2,0)[0])/4;
            float ratio = (Dxx+Dyy)*(Dxx+Dyy)/(Dxx*Dyy-Dxy*Dxy);            
            if (ratio > 10 || ratio < 0) {
                j--;
                continue;
            }

            //printf("Find descriptor %d..\n",j);

            // Find descriptor
            // Assign multiple descriptors if has multiple major orientations 
            vector<float> orientations;
            findOrientations(maxima[i], magPyramid, ornPyramid, &sigma, &orientations);
            for(int k=0;k<(int)orientations.size();k++) {
                corners.push_back(Feature(maxima[i], magPyramid, ornPyramid, &sigma, orientations[k]));
            }

            //printf("Features constructed\n");
        }

        printf("%u / %u features found\n", (unsigned int)corners.size(), (unsigned int)maxima.size());
    }

    // Once we have computed a digest for each of the images to align,
    // we can attempt to solve for the best alignment using RANSAC and
    // least squares
    Transform *align(Digest &other, Align::Mode m, int* inliers) {
        Transform *transform = NULL, *refined = NULL;
        if (m == Align::TRANSLATE) {
            transform = new Translation();
            refined   = new Translation();
        } else if (m == Align::SIMILARITY) {
            transform = new Similarity();
            refined   = new Similarity();
        } else if (m == Align::RIGID) {
            transform = new Rigid();
            refined   = new Rigid();
        } else if (m == Align::AFFINE) {
            transform = new Affine();
            refined   = new Affine();
        } else if (m == Align::PERSPECTIVE) {
            transform = new Perspective();
            refined   = new Perspective();
        } else {
            panic("Unknown transform type: %i\n", m);
        }

        // Associate the features with other features to produce
        // correspondences.
        vector<Correspondence> allCorrespondences, correspondences, tempCorr;

        for (unsigned i = 0; i < corners.size(); i++) {
            for (unsigned j = 0; j < other.corners.size(); j++) {
                allCorrespondences.push_back(Correspondence(&corners[i], &other.corners[j]));
            }
        }

        // Sort the correspondences by how good they are. Ones with a
        // low distance between their features will be at the start of
        // this list. This is a little inefficient, given that we're
        // going to throw out most of these, but compared to the image
        // processing steps, everything is cheap.
        ::std::sort(allCorrespondences.begin(), allCorrespondences.end());        

        // Select up to 1024 of the best correspondences.
        for (unsigned i = 0; i < allCorrespondences.size() && correspondences.size() < 1024; i++) {
            // No feature may be selected more than three times. If
            // you get a single image patch that matches everything,
            // it can make a big mess.
            if (allCorrespondences[i].a->usage < 3 &&
                allCorrespondences[i].b->usage < 3) {
                correspondences.push_back(allCorrespondences[i]);
                allCorrespondences[i].a->usage++;
                allCorrespondences[i].b->usage++;
            }
        }

        printf("%d correspondences found \n", (int)correspondences.size());
        /*
        // Print out the correspondences found for debugging.
        for (unsigned i = 0; i < correspondences.size(); i++) {
            printf("%f %f -> %f %f (%f)\n", 
                   correspondences[i].a->x,
                   correspondences[i].a->y,
                   correspondences[i].b->x,
                   correspondences[i].b->y,
                   correspondences[i].distance);
        }
        */

        // Run RANSAC
        int bestSeed = 0;
        float bestScore = 0;

        for (int iter = 0; iter < 50000; iter++) {
            // Reset the transform
            transform->reset();

            // Choose a random seed
            int seed = rand();
            srand(seed);

            // Pick the minimum number of correspondences required to generate a model
            for (int i = 0; i < transform->constraintsRequired(); i++) {
                int j = rand() % correspondences.size();
                transform->addCorrespondence(correspondences[j].a->x,
                                             correspondences[j].a->y,
                                             correspondences[j].b->x,
                                             correspondences[j].b->y);
            }

            // Do a least squares solve using the minimal number of constraints
            transform->solve();

            // Test the remaining correspondences against the model, counting the inliers
            float score = 0;
            for (unsigned i = 0; i < correspondences.size(); i++) {
                float x, y;
                transform->apply(correspondences[i].a->x,
                                 correspondences[i].a->y,
                                 &x, &y);
                x -= correspondences[i].b->x;
                y -= correspondences[i].b->y;

                // When does something count as an inlier? Using this
                // formula, a perfect match is 1, 1 pixel off is 0.5,
                // and it tails off with distance squared.
                score += 1.0/(x*x + y*y + 1);
            }

            // See if this is the best model we've found so far (highest number of inliers)
            if (score > bestScore) {
                bestScore = score;
                bestSeed = seed;
                //printf("%i %f\n", bestSeed, bestScore);
            }
            
            if (bestScore > transform->constraintsRequired()*20) break;

        }

        // Use the best seed we found again to compute its model
        transform->reset();
        srand(bestSeed);
        for (int i = 0; i < transform->constraintsRequired(); i++) {
            int j = rand() % correspondences.size();
            /*
            printf("Using constraint: %f %f -> %f %f\n",
                   correspondences[j].a->x,
                   correspondences[j].a->y,
                   correspondences[j].b->x,
                   correspondences[j].b->y);
            */
            transform->addCorrespondence(correspondences[j].a->x,
                                         correspondences[j].a->y,
                                         correspondences[j].b->x,
                                         correspondences[j].b->y);
        }
        transform->solve();        

        // Now we're going to throw in all the inliers under that
        // model into a single big least squares solve to refine the
        // solution.
        int numInliers=0;
        refined->reset();
        for (unsigned i = 0; i < correspondences.size(); i++) {
            float x, y;
            transform->apply(correspondences[i].a->x,
                             correspondences[i].a->y,
                             &x, &y);
            x -= correspondences[i].b->x;
            y -= correspondences[i].b->y;
            if (x*x + y*y < 2) {
                numInliers++;
                /*
                printf("Inlier: %f %f -> %f %f\n",
                       correspondences[i].a->x,
                       correspondences[i].a->y,
                       correspondences[i].b->x,
                       correspondences[i].b->y);
                */
                refined->addCorrespondence(correspondences[i].a->x,
                                           correspondences[i].a->y,
                                           correspondences[i].b->x,
                                           correspondences[i].b->y);
            }
        }
        refined->solve();        

        // Done! Return the refined solution.

        printf("%d inliers\n", numInliers);

        *inliers = numInliers;

        // Reset the usage counts in case we're going to reuse this digest
        for (size_t i = 0; i < corners.size(); i++) {
            corners[i].usage = 0;
        }

        for (size_t i = 0; i < other.corners.size(); i++) {
            other.corners[i].usage = 0;
        }

        delete transform;
        return refined;
    }

    // Visualize feature locations
    void displayFeatures(Window out) {

        for (int i=0;i<(int)corners.size(); i++) {
        
            int t, x, y;
            t = 0;
            x = (int)corners[i].x;
            y = (int)corners[i].y;
            
            for(int j=-4;j<=4;j++) {
                if(x+j >= 0 && x+j < out.width) {
                    out(x+j, y, t)[0] = 1;
                    out(x+j, y, t)[1] = 0;
                    out(x+j, y, t)[2] = 0;
                }
                if(y+j >= 0 && y+j < out.height) {
                    out(x, y+j, t)[0] = 1;
                    out(x, y+j, t)[1] = 0;
                    out(x, y+j, t)[2] = 0;
                }
            }
        }
    }

    vector<Feature> corners;

};



void Align::help() {
    printf("-align warps the top image on the stack to match the second image on the\n"
           "stack. align takes one argument, which must be \"translate\", \"similarity\", \n"
           "\"affine\", \"perspective\", or \"rigid\" and constrains the warp to be of that\n"
           "type.\n"
           "\n"
           "Usage: ImageStack -load a.jpg -load b.jpg -align similarity \\\n"
           "                  -add -save ab.jpg\n\n");
}

void Align::parse(vector<string> args) {
    assert(args.size() == 1, "-align takes one argument\n");

    Image result;

    if (args[0] == "translate") {
        result = apply(stack(1), stack(0), TRANSLATE);
    } else if (args[0] == "similarity") {
        result = apply(stack(1), stack(0), SIMILARITY);
    } else if (args[0] == "affine") {
        result = apply(stack(1), stack(0), AFFINE);
    } else if (args[0] == "rigid") {
        result = apply(stack(1), stack(0), RIGID);
    } else if (args[0] == "perspective") {
        result = apply(stack(1), stack(0), PERSPECTIVE);
    } else {
        panic("Unknown alignment type: %s. Must be translate, rigid, similarity, affine, or perspective.\n", args[0].c_str());
    }
    pop();
    push(result);
}


// Warp window b to match window a
Image Align::apply(Window a, Window b, Mode m) {

    // Iterative scale pyramid alignment
    #define SCALE_LEVELS 3

    Transform *transform = NULL;
    Transform *bestTransform = NULL;

    int score, bestScore=0;
    bool done = false;
    int indexA[] = {0,1,2,1,2,1,0,2,0};
    int indexB[] = {0,1,2,2,1,0,1,0,2};
    int thresh[] = {50,30,12,12,12,12,12,12,12};

    for(int i=0;i<SCALE_LEVELS*SCALE_LEVELS;i++) {
        
        if(done==true) {break;}

        int downA = 1 << indexA[i];
        int downB = 1 << indexB[i];

        //downA = 4; downB = 4;
        printf("scale (%d, %d)\n",indexA[i],indexB[i]);

        Image aa = Downsample::apply(a, downA, downA);
        Image bb = Downsample::apply(b, downB, downB);
        
        Digest digestA(aa);

        Digest digestB(bb);

        if (transform) delete transform;
        transform = digestA.align(digestB, m, &score);
        (*transform).adjustDownsampleScale(downA, downB);

        //done = true;

        if(score > bestScore){
            bestScore = score;
            if (bestTransform) delete bestTransform;
            bestTransform = transform;
            transform = NULL;
            if (score >= thresh[i]) break;
        }
    }

    Image out(a);
    for (int t = 0; t < out.frames; t++) {
        for (int y = 0; y < out.height; y++) {
            for (int x = 0; x < out.width; x++) {
                float fx, fy;
                bestTransform->apply(x, y, &fx, &fy);
                b.sample2D(fx, fy, t, out(x, y, t));
            }
        }
    }

    if (bestTransform) delete bestTransform;
    if (transform) delete transform;

    return out;

}


void AlignFrames::help() {

}

void AlignFrames::parse(vector<string> args) {
    assert(args.size() == 1, "-alignframes takes one argument\n");

    if (args[0] == "translate") {
        apply(stack(0), Align::TRANSLATE);
    } else if (args[0] == "similarity") {
        apply(stack(0), Align::SIMILARITY);
    } else if (args[0] == "affine") {
        apply(stack(0), Align::AFFINE);
    } else if (args[0] == "rigid") {
        apply(stack(0), Align::RIGID);
    } else if (args[0] == "perspective") {
        apply(stack(0), Align::PERSPECTIVE);
    } else {
        panic("Unknown alignment type: %s. Must be translate, rigid, similarity, affine, or perspective.\n", args[0].c_str());
    }
}

void AlignFrames::apply(Window im, Align::Mode m) {

    assert(im.frames > 1, "Input must have at least two frames\n");

    // make a digest for each input frame

    vector<Digest *> digests;
    map<pair<int, int>, Transform *> transforms;

    printf("Extracting features...\n");
    for (int t = 0; t < im.frames; t++) {
        digests.push_back(new Digest(Window(im, 0, 0, t, 
                                            im.width, im.height, 1)));

    }
    
    printf("Matching features...\n");

    float bestScore = 0;
    int bestT = 0;

    for (int t1 = 0; t1 < im.frames; t1++) {
        printf("Aligning everything to frame %d\n", t1);
        float score = 100000;
        for (int t2 = 0; t2 < im.frames; t2++) {
            if (t1 == t2) continue;

            int inliers = 0;
            Transform *t = digests[t1]->align(*digests[t2], m, &inliers);

            if (inliers < score) score = inliers;

            transforms[make_pair(t1, t2)] = t;            

            if (score < bestScore) break;
        }

        printf("\nScore %d = %f\n\n", t1, score);
        if (score > bestScore) {
            bestScore = score;
            bestT = t1;
        }
    }

    // We did the best when we aligned everything to frame bestT        
    //Image out(im);

    printf("Warping");
    for (int t = 0; t < im.frames; t++) {
        printf("."); fflush(stdout);
        if (t == bestT) continue;
        Image tmp = Image(Window(im, 0, 0, t, im.width, im.height, 1));

        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                float fx, fy;
                Transform *trans = transforms[make_pair(bestT, t)];
                trans->apply(x, y, &fx, &fy);
                tmp.sample2D(fx, fy, 0, im(x, y, t));
            }
        }
    }
    printf("\n");
    
    //Display::apply(out);

    for (size_t i = 0; i < digests.size(); i++) {
        delete digests[i];
    }

    for (int t1 = 0; t1 < im.frames; t1++) {
        for (int t2 = 0; t2 < im.frames; t2++) {
            if (t1 == t2) continue;
            delete transforms[make_pair(t1, t2)];
        }
    }
}

#include "footer.h"
