#include "main.h"
#include "LAHBPCG.h"
#include "File.h"
#include "Calculus.h"
#include "Arithmetic.h"
#include "Convolve.h"
#include <list>
#include "header.h"

// This is an implementation of "Locally Adaptive Hierachical Basis
// Preconditioning" by Rick Szeliski. It was written by Eric Chu with
// considerable help and instruction from Rick Szeliski and Dani
// Lichinszki. In particular, this code was adapted from Dani's matlab
// implementation and "optimized" for images (instead of sparse
// vectors). The code was slightly modified by Andrew Adams for
// inclusion in ImageStack.

class PCG {
    // Four off-diagonal elements
    struct S_elems {
        float SS;
        float SE;
        float SN;
        float SW;
    };
public:
    PCG(Window d, Window gx, Window gy, Window w, Window sx, Window sy) 
        : AW(d.width, d.height, 1, 1),
          AN(d.width, d.height, 1, 1),
          w(w),
          sx(sx),
          sy(sy),
          b(d.width, d.height, 1, d.channels), 
          f(d.width, d.height, 1, d.channels), 
          hbRes(d.width, d.height, 1, d.channels), 
          AD(d.width, d.height, 1, 1), 
          max_length(d.width*d.height)
        {
            assert(d.frames == 1 && gx.frames == 1 && gy.frames == 1 
                   && w.frames == 1 && sx.frames == 1 && sy.frames == 1,
                   "should run PCG on single frame at a time!");
            
            assert(w.channels == 1 && sx.channels == 1 && sy.channels == 1,
                   "Weights must be single-channel!");

            float sub_y, sub_x;
            float add_y, add_x;
            for (int t = 0; t < b.frames; t++) {
                for (int y = 0; y < b.height; y++) {
                    for (int x = 0; x < b.width; x++) {

                        if (y == b.height-1) {
                            add_y = 0;
                        } else {
                            add_y = sy(x,y+1,t)[0];
                        }
                        
                        if (x == b.width-1) {
                            add_x = 0;
                        } else {
                            add_x = sx(x+1,y,t)[0];
                        }
                        
                        AD(x,y,t)[0] = sx(x,y,t)[0] + add_x + w(x,y,t)[0] + sy(x,y,t)[0] + add_y;
                        
                        if (y == sy.height-1) {
                            AN(x,y,t)[0] = 0;
                        } else {
                            AN(x,y,t)[0] = -sy(x,y+1)[0];
                        }
                        
                        if (x == sx.width-1) {
                            AW(x,y,t)[0] = 0;
                        } else {
                            AW(x,y,t)[0] = -sx(x+1,y)[0];
                        }

                        for (int c = 0; c < b.channels; c++) {
                            if (y == b.height-1) {
                                sub_y = 0;
                            } else {
                                sub_y = gy(x,y+1,t)[c]*sy(x,y+1,t)[0];
                            }
            
                            if (x == b.width-1) {
                                sub_x = 0;
                            } else {
                                sub_x = gx(x+1,y,t)[c]*sx(x+1,y,t)[0];
                            }
            
                            b(x,y,t)[c] = (gy(x,y,t)[c]*sy(x,y,t)[0] - sub_y + 
                                           gx(x,y,t)[c]*sx(x,y,t)[0] - sub_x 
                                           + w(x,y,t)[0]*d(x,y,t)[c]);
                        }
                    }
                }
            }
            // set up indices
            RBBmaps();
            // compute preconditioner... 
            constructPreconditioner();
        }
  
    void solve(Window guess, int max_iter, float tol);

private:
    Image Ax(Window im); // apply A to "x" the image

    Image hbPrecondition(Image r); // apply the preconditioner to the residual r
  
    float dot(Window a, Window b);

    void alphax(float alpha, Window im);
  
    void RBBmaps();
    void constructPreconditioner();
    void ind2xy(const unsigned int index, int & x, int & y);
  
    inline unsigned int varIndices(const int x, const int y)
        {
            if (x < 0 || x >= f.width)
                return max_length;
            if (y < 0 || y >= f.height)
                return max_length;
    
            return (x*f.height + y);
        }
  
    // these are just references to already allocated memory
    //Image ADcoarse; 
    Image AW;
    Image AN;
    Window w;
    Window sx;
    Window sy;
  
    Image b; // const?
  
    Image f; // current iterate storate....
    Image hbRes; 
  
    Image AD; // diagonalized A matrix
    const unsigned int max_length;
  
    vector< vector<unsigned int> > index_map; // goes up to 2^32
    vector< vector< S_elems > > S; // 4 channel images that store S weights...
};

// computes the vector indices into an image (0 to m*n - 1)
void PCG::RBBmaps() {
    int numOctaves = ceil(logf(min(f.width, f.height))/logf(2.0f));
    int by = 0; int bx = 0; // (0,0) is black
  
    int a = 1;
    vector<unsigned int> valid;
    vector<unsigned int> newValid;

    for (int i = 0; i < numOctaves; i++) {

        vector<unsigned int> indices1;
        vector<unsigned int> indices2;

        //indices1.reserve(f.width * f.height);
        //indices2.reserve(f.width * f.height);

        if (valid.empty()) { // loop over all elements
            for (int x = 0; x < f.width; x++) {
                for (int y = 0; y < f.height; y++) {
          
                    if ((x + y) % (2*a) == (by + bx + a) % (2*a)) {
                        indices1.push_back(x * f.height + y);
                    } else {
                        valid.push_back(x * f.height + y);
                    }
            
                }
            }

            if (indices1.size() == 0) {
                break;
            }
            index_map.push_back(indices1);
      
            int x,y;

            newValid.clear();
            for (vector<unsigned int>::iterator it = valid.begin(); it != valid.end(); ++it) {
                ind2xy(*it, x, y);
                if (y % (2*a) == (by + a) % (2*a)) {
                    indices2.push_back(*it);
                } else {
                    newValid.push_back(*it);
                }
            }
            valid.swap(newValid);

            if (indices2.size() == 0) {
                break;
            }
            index_map.push_back(indices2);

        } else { // iterate over list elements (containing valid indices)
            int x,y;
        
            newValid.clear();
            for (vector<unsigned int>::iterator it = valid.begin(); it != valid.end(); ++it) {
                ind2xy(*it, x, y);
                if ((x + y) % (2*a) == (bx + by + a) % (2*a)) {
                    indices1.push_back(*it);
                } else {
                    newValid.push_back(*it);
                }
            }
            valid.swap(newValid);

            if (indices1.size() == 0) {
                break;
            }
            index_map.push_back(indices1);

            newValid.clear();      
            for (vector<unsigned int>::iterator it = valid.begin(); it != valid.end(); ++it) {
                ind2xy(*it, x, y);
                if ((y) % (2*a) == (by + a) % (2*a)) {
                    indices2.push_back(*it);
                } else {
                    newValid.push_back(*it);
                }
            }
            valid.swap(newValid);

            if (indices2.size() == 0) {
                break;
            }
            index_map.push_back(indices2);
        }
    
        a *= 2;
    }
}

// computes the x,y index from a vector index
void PCG::ind2xy(const unsigned int index, int & x, int & y) {
    x = index / f.height;
    y = index % f.height;
}

// computes the preconditioner
void PCG::constructPreconditioner() {
    // this is a giant function....... can we "split" it?
    assert(!index_map.empty(), "computePreconditioner() needs to run after RBBmaps()");
  
    int x,y, x1,y1;
    // fill out S matrix
    for (int k = 0; k < (int) index_map.size(); k++) {
  
        bool oddLevel = (k+1) % 2;
        int stride = 1 << (k/2);
        //printf("stride: %d\n", stride);
    
        unsigned int dn1, dn2;
        if (oddLevel) {
            dn1 = stride; dn2 = stride*f.height;
        } else {
            dn1 = stride*(f.height - 1);
            dn2 = stride*(f.height + 1);
        }
    
        // compute S elements at this level
        vector< S_elems > S_elem_vec;
        vector< float > AD_old; // retain old values of AD

        //printf("dn1 %d, dn2 %d\n", dn1, dn2);
        S_elems elems;// = {0, 0, 0, 0};
        // on this level, we use the indices in index_map[k]        
        for (vector<unsigned int>::iterator idx = index_map[k].begin();
             idx != index_map[k].end(); ++idx) {
            //int x, y, x1, y1;// x2, y2;
      
            ind2xy(*idx, x, y);
      
            elems.SS = -AN(x,y)[0]/(AD(x,y)[0]); // SS
            elems.SE = -AW(x,y)[0]/(AD(x,y)[0]); // SE

            if (*idx < dn1) { // *idx - dn1 < 0
                ind2xy(max_length + (*idx) - dn1, x1, y1);
            } else {
                ind2xy((*idx - dn1) % max_length, x1, y1);
            }
            elems.SN = -AN(x1,y1)[0]/(AD(x,y)[0]); // SN

            if (*idx < dn2) { // *idx - dn2 < 0
                ind2xy(max_length + (*idx) - dn2, x1, y1);
            } else {
                ind2xy((*idx - dn2) % max_length, x1, y1);
            }
            elems.SW = -AW(x1,y1)[0]/(AD(x,y)[0]); // SW

            S_elem_vec.push_back(elems);
            AD_old.push_back(AD(x,y)[0]);
            //AN(x1,y1)*elems.SN
            //AW(x2,y2)*elems.SW
            //printf("ind: %d, SW: %f\n", *idx, elems.SW);
        } /* end vector iterator */
        S.push_back(S_elem_vec);
    
        // now we need to redistribute edges....
        // need temp storage for this new AN....
        Image AN_tmp(AN.width, AN.height, AN.frames, AN.channels); // actually "sparse"
        Image AW_tmp(AW.width, AW.height, AW.frames, AW.channels); // actually "sparse"
  
        int i = 0;
        // modify AD at this level
        for (vector<unsigned int>::iterator idx = index_map[k].begin();
             idx != index_map[k].end(); ++idx) {
        
            S_elems elems = S_elem_vec[i];
            ind2xy(*idx, x, y);
      
            ind2xy((*idx - dn1) % max_length, x1, y1);
            AD(x1,y1)[0] += AN(x1,y1)[0]*elems.SN;
      
            ind2xy((*idx - dn2) % max_length, x1, y1);
            AD(x1,y1)[0] += AW(x1,y1)[0]*elems.SW;
      
            ind2xy((*idx + dn1) % max_length, x1, y1);
            AD(x1,y1)[0] += AN(x,y)[0]*elems.SS;
      
            ind2xy((*idx + dn2) % max_length, x1, y1); 
            AD(x1,y1)[0] += AW(x,y)[0]*elems.SE;
      
            /* end modify AD */
      
            /* now eliminate connections */
            unsigned int n_ind, w_ind, s_ind, e_ind;
            if (oddLevel) {
                n_ind = varIndices(x, y-stride);
                w_ind = varIndices(x-stride, y);
                s_ind = varIndices(x, y+stride);
                e_ind = varIndices(x+stride, y);
            } else {
                n_ind = varIndices(x-stride, y-stride);
                w_ind = varIndices(x-stride, y+stride);
                s_ind = varIndices(x+stride, y+stride);
                e_ind = varIndices(x+stride, y-stride);
            }
            //printf("n: %d, w: %d, s: %d, e: %d\n", n_ind, w_ind, s_ind, e_ind);
      
            // eliminate NS connections
            bool ns = false;
            float ns_weight = 0.0f;
            int n_x, n_y, s_x, s_y;
      
            ind2xy(n_ind, n_x, n_y);
            ind2xy(s_ind, s_x, s_y);
            if (n_ind < max_length && s_ind < max_length) {
                ns = true;
                if (oddLevel) {
                    ns_weight = -AD_old[i]*elems.SN*elems.SS;
                } else {
                    ns_weight = -AD_old[i]*elems.SW*elems.SE;
                }
        
                AD(n_x, n_y)[0] += ns_weight;
                AD(s_x, s_y)[0] += ns_weight;
                //printf("ns_weight: %f\n", ns_weight);
            }
      
            // eliminate WE connections
            bool we = false;
            float we_weight = 0.0f;
            int w_x, w_y, e_x, e_y;
      
            ind2xy(w_ind, w_x, w_y);
            ind2xy(e_ind, e_x, e_y);
            if (w_ind < max_length && e_ind < max_length) {
                we = true;
                if (oddLevel) {
                    we_weight = -AD_old[i]*elems.SW*elems.SE;
                } else {
                    we_weight = -AD_old[i]*elems.SN*elems.SS;
                }
        
                AD(w_x, w_y)[0] += we_weight;
                AD(e_x, e_y)[0] += we_weight;
                //printf("we_weight: %f\n", we_weight);
            }
      
            // redistribute "connected" edges / weights
            float nw_weight, ws_weight, se_weight, en_weight;
            nw_weight = ws_weight = se_weight = en_weight = 0.0f;
            if (oddLevel) {
                if (n_ind < max_length && w_ind < max_length) {
                    nw_weight = AD_old[i]*elems.SN*elems.SW;
                    AN_tmp(w_x, w_y)[0] -= nw_weight; 
                }
                if (w_ind < max_length && s_ind < max_length) {
                    ws_weight = AD_old[i]*elems.SW*elems.SS;
                    AW_tmp(w_x, w_y)[0] -= ws_weight; 
                    //printf("nw_weight: %f\n", nw_weight);
                }
                if (e_ind < max_length && s_ind < max_length) {
                    se_weight = AD_old[i]*elems.SE*elems.SS;
                    AN_tmp(s_x, s_y)[0] -= se_weight; 
                    //printf("nw_weight: %f\n", nw_weight);
                }
                if (e_ind < max_length && n_ind < max_length) {
                    en_weight = AD_old[i]*elems.SE*elems.SN;
                    AW_tmp(n_x, n_y)[0] -= en_weight; 
                    //printf("nw_weight: %f\n", nw_weight);
                }
      
            } else { 
                if (n_ind < max_length && w_ind < max_length) {
                    nw_weight = AD_old[i]*elems.SN*elems.SW;
                    AN_tmp(n_x, n_y)[0] -= nw_weight; 
                    //printf("n_ind: %d, ind: %d, ni_weight: %f, wi_weight: %f\n", n_ind, *idx, elems.SW, elems.SN); // how to index into S...?

                    //printf("nw_weight: %f\n", nw_weight);
                }
                if (w_ind < max_length && s_ind < max_length) {
                    ws_weight = AD_old[i]*elems.SN*elems.SE;
                    AW_tmp(w_x, w_y)[0] -= ws_weight; 
                    //printf("nw_weight: %f\n", nw_weight);
                }
                if (e_ind < max_length && s_ind < max_length) {
                    se_weight = AD_old[i]*elems.SE*elems.SS;
                    AN_tmp(e_x, e_y)[0] -= se_weight; 
                    //printf("nw_weight: %f\n", nw_weight);
                }
                if (e_ind < max_length && n_ind < max_length) {
                    en_weight = AD_old[i]*elems.SS*elems.SW;
                    AW_tmp(n_x, n_y)[0] -= en_weight; 
                    //printf("nw_weight: %f\n", nw_weight);
                }
            }
      
            // normalize the redistributed weights
            if (ns || we) {
                float total = nw_weight + ws_weight + se_weight + en_weight;
                if (total != 0) {
                    nw_weight /= total;
                    ws_weight /= total;
                    se_weight /= total;
                    en_weight /= total;
                }
        
                // now, redistribute
                static const float sN = 2;
                float distWeight = sN*(ns_weight + we_weight);
        
                //printf("nw_weight %f, ws_weight %f, se_weight %f, en_weight %f, distWeight %f\n", nw_weight, ws_weight, se_weight, en_weight, distWeight);
                if (oddLevel) {
                    if (n_ind < max_length && w_ind < max_length) {
                        AN_tmp(w_x, w_y)[0] += nw_weight*distWeight; 
                    }
                    if (w_ind < max_length && s_ind < max_length) {
                        AW_tmp(w_x, w_y)[0] += ws_weight*distWeight; 
                    }
                    if (e_ind < max_length && s_ind < max_length) {
                        AN_tmp(s_x, s_y)[0] += se_weight*distWeight; 
                    }
                    if (e_ind < max_length && n_ind < max_length) {
                        AW_tmp(n_x, n_y)[0] += en_weight*distWeight; 
                    }
                } else {
                    if (n_ind < max_length && w_ind < max_length) {
                        AN_tmp(n_x, n_y)[0] += nw_weight*distWeight; 
                    }
                    if (w_ind < max_length && s_ind < max_length) {
                        AW_tmp(w_x, w_y)[0] += ws_weight*distWeight; 
                    }
                    if (e_ind < max_length && s_ind < max_length) {
                        AN_tmp(e_x, e_y)[0] += se_weight*distWeight; 
                    }
                    if (e_ind < max_length && n_ind < max_length) {
                        AW_tmp(n_x, n_y)[0] += en_weight*distWeight; 
                    }
                }
        
                if (n_ind < max_length && w_ind < max_length) {
                    AD(n_x, n_y)[0] -= nw_weight*distWeight;
                    AD(w_x, w_y)[0] -= nw_weight*distWeight; 
                }
                if (w_ind < max_length && s_ind < max_length) {
                    AD(w_x, w_y)[0] -= ws_weight*distWeight; 
                    AD(s_x, s_y)[0] -= ws_weight*distWeight;
                }
                if (e_ind < max_length && s_ind < max_length) {
                    AD(s_x, s_y)[0] -= se_weight*distWeight;
                    AD(e_x, e_y)[0] -= se_weight*distWeight; 
                }
                if (e_ind < max_length && n_ind < max_length) {
                    AD(n_x, n_y)[0] -= en_weight*distWeight; 
                    AD(e_x, e_y)[0] -= en_weight*distWeight; 
                }
        
            }

            i++; // go to next element to "eliminate"

        } /* end vector iterator */
    
        AN = AN_tmp;
        AW = AW_tmp;
    } /* end for all levels */
  
}

// applies the sparse, pentadiagonal matrix A to x (stored in im)
// assumes gradient images taken from ImageStack's gradient operator
// (i.e. backward differences) if not, results could be bogus!
Image PCG::Ax(Window im) {
    float a1,a2,a3;

    // (Ax + w)* x
    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            int x = 0;

            a1 = 0;
            a2 = sx(x,y,t)[0] + sx(x+1,y,t)[0]+ w(x,y,t)[0];
            a3 = -sx(x+1,y,t)[0];
            for (int c = 0; c < im.channels; c++) {
                f(x,y,t)[c] = a2*im(x,y,t)[c] + a3*im(x+1,y,t)[c];
            }

            for (x = 1; x < im.width-1; x++) {               
                a1 = -sx(x,y,t)[0];
                a2 = sx(x,y,t)[0] + sx(x+1,y,t)[0] + w(x,y,t)[0]; // a_(ij)x_(ij) + w_(ij)*x_(ij)
                a3 = -sx(x+1,y,t)[0]; // AW
                for (int c = 0; c < im.channels; c++) {
                    f(x,y,t)[c] = a1*im(x-1,y,t)[c] + a2*im(x,y,t)[c] + a3*im(x+1,y,t)[c]; 
                }
            }

            x = im.width-1;
            a1 = -sx(x,y,t)[0];
            a2 = sx(x,y,t)[0] + w(x,y,t)[0]; 
            a3 = 0; //AW
            for (int c = 0; c < im.channels; c++) {
                f(x,y,t)[c] = a1*im(x-1,y,t)[c] + a2*im(x,y,t)[c];
            }
        }
    }
  
    // Ay*x + (Ax + w)*x
    for (int t = 0; t < im.frames; t++) {
        // for cache coherency
        for (int x1 = 0; x1 < im.width; x1+= 8) {

            int y = 0;
            for (int x = x1; (x < x1 + 8) && (x < im.width); x++) {
                a1 = 0;
                a2 = sy(x,y,t)[0] + sy(x,y+1,t)[0];
                a3 = -sy(x,y+1,t)[0]; //AN
                for (int c = 0; c < im.channels; c++) {
                    f(x,y,t)[c] += a2*im(x,y,t)[c] + a3*im(x,y+1,t)[c];
                }
            }

            for (y = 1; y < im.height-1; y++) {
                for (int x = x1; (x < x1 + 8) && (x < im.width); x++) {

                    a1 = -sy(x,y,t)[0];
                    a2 = sy(x,y,t)[0] + sy(x,y+1,t)[0];
                    a3 = -sy(x,y+1,t)[0];
                    for (int c = 0; c < im.channels; c++) {
                        f(x,y,t)[c] += a1*im(x,y-1,t)[c] + a2*im(x,y,t)[c] + a3*im(x,y+1,t)[c];
                    }
                }
            }

            y = im.height-1;
            for (int x = x1; (x < x1 + 8) && (x < im.width); x++) {
                a1 = -sy(x,y,t)[0];
                a2 = sy(x,y,t)[0]; 
                a3 = 0; //AN
                for (int c = 0; c < im.channels; c++) {
                    f(x,y,t)[c] += a1*im(x,y-1,t)[c] + a2*im(x,y,t)[c];
                }
            }
        }
    }

    return f;
}

// apply the preconditioner to the residual r
Image PCG::hbPrecondition(Image r) {
    // wonder if there's a way to apply the preconditioner in a cache coherent manner....
    hbRes = r.copy(); // ugh.. another deep copy... (no way around it...)
    int x,y,x1,y1;
    for (int k = 0; k < (int) index_map.size(); k ++) {
        //int i = 0;
        // turns out that recomputing these numbers is 
        // faster than accessing it in memory because of the misses
        bool oddLevel = (k+1) % 2;
        int stride = 1 << (k/2);
        //printf("stride: %d\n", stride);
    
        unsigned int dn1, dn2;
        if (oddLevel) {
            dn1 = stride; dn2 = stride*f.height;
        } else {
            dn1 = stride*(f.height - 1);
            dn2 = stride*(f.height + 1);
        }
    
        int i = 0;
        // S'*d
        for (vector<unsigned int>::iterator idx = index_map[k].begin();
             idx != index_map[k].end(); ++idx) {
            // since this is S'*d, the index map represents "column" now
            //int x, y, x1, y1, x2, y2;
            S_elems elems = S[k][i];
  
            ind2xy(*idx, x, y);
      
            if (*idx + dn1 < max_length) {
                //elems.SS = 0;
                ind2xy((*idx + dn1) % max_length, x1, y1);
                //y1 %= sy.height; x1 %= sy.width;
                for (int c = 0; c < hbRes.channels; c++) {
                    hbRes(x1, y1)[c] += hbRes(x, y)[c]*elems.SS;
                }
            }
      
            if (*idx + dn2 < max_length) {
                ind2xy((*idx + dn2) % max_length, x1, y1);
                //y1 %= sx.height; x1 %= sx.width;
                for (int c = 0; c < hbRes.channels; c++) {
                    hbRes(x1, y1)[c] += hbRes(x, y)[c]*elems.SE;
                }
            }
      
            if (*idx >= dn1) { // *idx - dn1 >= 0
                ind2xy((*idx - dn1) % max_length, x1, y1);
                //y1 %= sy.height; x1 %= sy.width;
                for (int c = 0; c < hbRes.channels; c++) {
                    hbRes(x1, y1)[c] += hbRes(x, y)[c]*elems.SN;
                }
            } 
      
            if (*idx >= dn2) { // *idx - dn2 >= 0
                ind2xy((*idx - dn2) % max_length, x1, y1);
                //y1 %= sx.height; x1 %= sx.width;
                for (int c = 0; c < hbRes.channels; c++) {
                    hbRes(x1, y1)[c] += hbRes(x, y)[c]*elems.SW;        
                }
            }
      
            i++;
        }
    }
  
    Divide::apply(hbRes, AD); // invert the diagonal

    // lowest level is identity matrix so it's ignored (not even stored in index_map)
    for (int k = (int) index_map.size() - 1; k >= 0; k--) {
        //int i = index_map.size() - 1;
        bool oddLevel = (k+1) % 2;
        int stride = 1 << (k/2);
        //printf("stride: %d\n", stride);
    
        unsigned int dn1, dn2;
        if (oddLevel) {
            dn1 = stride;
            dn2 = stride*f.height;
        } else {
            dn1 = stride*(f.height - 1);
            dn2 = stride*(f.height + 1);
        }
    
        int i = 0;
        //S*d
        for (vector<unsigned int>::iterator idx = index_map[k].begin();
             idx != index_map[k].end(); ++idx) {
            // since this is S*d, the index map represents "row" now
            //int x, y, x1, y1, x2, y2;
            S_elems elems = S[k][i];
      
            ind2xy(*idx, x, y);
      
            if (*idx + dn1 < max_length) {
                ind2xy((*idx + dn1) % max_length, x1, y1);
                //y1 %= sy.height; x1 %= sy.width;
                for (int c = 0; c < hbRes.channels; c++) {
                    hbRes(x, y)[c] += hbRes(x1, y1)[c]*elems.SS;
                }
            }
      
            if (*idx + dn2 < max_length) {
                ind2xy((*idx + dn2) % max_length, x1, y1);
                //y1 %= sx.height; x1 %= sx.width;
                for (int c = 0; c < hbRes.channels; c++) {
                    hbRes(x, y)[c] += hbRes(x1, y1)[c]*elems.SE;
                }
            }
      
            if (*idx >= dn1) { // *idx - dn1 >= 0
                ind2xy((*idx - dn1) % max_length, x1, y1);
                //y1 %= sy.height; x1 %= sy.width;
                for (int c = 0; c < hbRes.channels; c++) {
                    hbRes(x, y)[c] += hbRes(x1, y1)[c]*elems.SN;
                }
            } 
      
            if (*idx >= dn2) { // *idx - dn2 >= 0
                ind2xy((*idx - dn2) % max_length, x1, y1);
                //y1 %= sx.height; x1 %= sx.width;
                for (int c = 0; c < hbRes.channels; c++) {
                    hbRes(x, y)[c] += hbRes(x1, y1)[c]*elems.SW;
                }
            }
      
            i++;
        }
        //i--;
    }

    return hbRes;
}

// compute dot product
float PCG::dot(Window a, Window b) {
    assert(a.frames == b.frames && a.height == b.height && a.width == b.width && a.channels == b.channels,
           "a and b need to be the same size\n");
    double result = 0;
    for (int t = 0; t < a.frames; t++) {
        for (int y = 0; y < a.height; y++) {
            for (int x = 0; x < a.width; x++) {
                for (int c = 0; c < a.channels; c++) {
                    result += a(x,y,t)[c]*b(x,y,t)[c];
                }
            }
        }
    }
    return result;
}

// solve the PCG!
void PCG::solve(Window guess, int max_iter, float tol) {

    Image dr_tmp, s;

    Image r(b); // we currently do not use b anywhere else, so i reuse its memory
    Subtract::apply(r,Ax(guess));
    Image dr = hbPrecondition(r); // precondition, dr to differentiate from d

    float delta = dot(r,dr);
    float epsilon = tol*tol*delta;
    printf("initial error: %f\n", delta);
  
    for (int i = 1; i <= max_iter; i++) {
        if (delta < epsilon) {
            break;
        }

        Image wr = Ax(dr);
        float alpha = delta / dot(dr,wr);

        dr_tmp = dr.copy();
    
        Scale::apply(dr, alpha);
        Add::apply(guess, dr);    // guess = guess + alpha*dr
        Scale::apply(wr, alpha);
        Subtract::apply(r, wr);   // r = r - alpha*wr
    
        float resNorm = dot(r,r);
        printf("iteration %d, error %f\n", i, resNorm);
        if (resNorm < epsilon) {
            break;
        }
    
        s = hbPrecondition(r);    // precondition
        float delta_old = delta;
        delta = dot(r,s);
        float beta = delta / delta_old;
    
        Scale::apply(dr_tmp, beta);    
        dr = s;
        Add::apply(dr,dr_tmp);
    }
}

void LAHBPCG::help() {
    pprintf("-lahbpcg takes six images from the stack and treats them as a target"
            " output, x gradient, and y gradient, and then the respective weights"
            " for each term. The weights may be single-channel or have the same"
            " number of channels as the target images. It then attempts to solve for"
            " the image which best achieves that target ouput and those target"
            " gradients in the weighted-least-squares sense using a preconditioned"
            " weighted least squares solver. This technique is useful for a variety"
            " of problems with constraints expressed in the gradient domain,"
            " including Poisson solves, making a sparse labelling dense, and other"
            " gradient-domain techniques. The problem formulation is from Pravin"
            " Bhat's \"GradientShop\", and"
            " the preconditioner is the Locally Adapted Hierarchical Basis"
            " Preconditioner described by Richard Szeliski.\n"
            "\n"
            "This operator takes two arguments. The first specifies the maximum"
            " number of iterations, and the second specifies the error required for"
            " convergence\n"
            "\n"
            "The following example takes a sparse labelling of an image im.jpg, and"
            " expands it to be dense in a manner that respects the boundaries of"
            " the image. The target image is the labelling, with weights indicating"
            " where it is defined. The target gradients are zero, with weights"
            " inversely proportional to gradient strength in the original image."
            "\n"
            "Usage: ImageStack -load sparse_labels.tmp \\\n"
            "                  -push -dup \\\n"
            "                  -load sparse_weights.tmp \\\n"
            "                  -load im.jpg -gradient x -colorconvert rgb gray \\\n"
            "                  -eval \"1/(100*val^2+0.001)\" \\\n"
            "                  -load im.jpg -gradient y -colorconvert rgb gray \\\n"
            "                  -eval \"1/(100*val^2+0.001)\" \\\n"
            "                  -lahbpcg 5 0.001 -save out.png\n");
}

void LAHBPCG::parse(vector<string> args) {
    assert(args.size() == 2, "-lahbpcg takes two arguments\n");
  
    Image result;
  
    result = apply(stack(5), stack(4), stack(3), stack(2), stack(1), stack(0), readInt(args[0]), readFloat(args[1]));
  
    for (int i = 0; i < 5; i ++) {
        pop();
    }
    push(result);
}

Image LAHBPCG::apply(Window d, Window gx, Window gy, Window w, Window sx, Window sy, int max_iter, float tol)
{
    // check to make sure have same # of frames and same # of channels
    // assumes gradient images computed using ImageStack's gradient, which is
    // slightly different from the standard convolution gradient
    /*Image gx1(d);
      Image gy1(d);
      Gradient::apply(gx1, 'x');
      Gradient::apply(gy1, 'y');*/
  
    // runtime is LINEAR in the number of pixels!
  
    // solve the problem
    // minimize
    //  sum_(i,j) w_(i,j)*(f_(i,j) - d_(i,j))^2 + sum_(i,j) sx_i (f_(i+1,j) - f_(i,j) - gx(i,j))^2
    //    + sy_j (f_(i,j+1) - f(i,j) - gy(i,j))^2
    // which can be written as
    //
    // minimize x^T A x + b^T x + c
    //
    // the problem reduces to solving Ax = b
    assert(max_iter >= 0, "maximum number of iterations should be nonnegative\n");
    assert(tol < 1, "tolerance should be less than 1\n");
  
    assert(d.frames == gx.frames && d.frames == gy.frames && d.frames == w.frames &&
           d.frames == sx.frames && d.frames == sy.frames, "requires input images to have same number of frames\n");
  
    assert(d.width == gx.width && d.width == gy.width && d.width == w.width &&
           d.width == sx.width && d.width == sy.width, "requires input images to have same width\n");
  
    assert(d.height == gx.height && d.height == gy.height && d.height == w.height &&
           d.height == sx.height && d.height == sy.height, "requires input images to have same height\n");
  
    assert(d.channels == gx.channels && d.channels == gy.channels && 
           (w.channels == 1) && (sx.channels == 1) && (sy.channels == 1),
           "Image and gradients must have a matching number of channels. Weight terms must have one channel.\n");
  
  
    Image out(d.width, d.height, d.frames, d.channels);
  
    // solves frames independently
    for (int t = 0; t < d.frames; t++) {
        Window dw(d, 0, 0, t, d.width, d.height, 1);
        Window gxw(gx, 0, 0, t, gx.width, gx.height, 1);
        Window gyw(gy, 0, 0, t, gy.width, gy.height, 1);
        Window ww(w, 0, 0, t, w.width, w.height, 1);
        Window sxw(sx, 0, 0, t, sx.width, sx.height, 1);
        Window syw(sy, 0, 0, t, sy.width, sy.height, 1);
      
        Window outw(out, 0, 0, t, out.width, out.height, 1);

        printf("Computing preconditioner...\n");
        PCG solver(dw, gxw, gyw, ww, sxw, syw);  

        printf("Solving...\n");
        solver.solve(outw, max_iter, tol);
    }

    return out;
}
#include "footer.h"
