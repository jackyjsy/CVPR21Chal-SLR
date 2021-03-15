#ifndef IMAGESTACK_EIGENVECTORS_H
#define IMAGESTACK_EIGENVECTORS_H

#include <math.h>
#include "header.h"

class Eigenvectors {
  public:
    Eigenvectors(int in_dimensions, int out_dimensions) {
        d_in = in_dimensions;
        d_out = out_dimensions;
        
        covariance = new double[d_in*d_in];
        mean = new double[d_in];
        eigenvectors = new double[d_in*d_out];
        tmp = new double[d_in*d_out];
        computed = false;
        for (int i = 0; i < d_in; i++) {
            mean[i] = 0;
            for (int j = 0; j < d_in; j++) {
                covariance[i*d_in + j] = 0;                
                if (j < d_out) {
                    eigenvectors[i*d_out + j] = 0;
                    tmp[i*d_out + j] = 0;
                }
            }
        }
        count = 0;
    }

    void add(const float *v) {
        for (int i = 0; i < d_in; i++) {
            for (int j = 0; j < d_in; j++) {
                covariance[i*d_in+j] += v[i]*v[j];
            }
            mean[i] += v[i];
        }
        count++;
    }

    // how much of each eigenvector is in a particular vector?
    // multiply the vector by the transpose of the eigenvector matrix
    void apply(const float *v_in, float *v_out) {
        if (!computed) compute();
        
        for (int i = 0; i < d_out; i++) {
            v_out[i] = 0;
            for (int j = 0; j < d_in; j++) {
                v_out[i] += eigenvectors[j*d_out+i] * v_in[j];
            }
        }
    }

    // Get the nth eigenvector
    void getEigenvector(int idx, float *v_out) {
        for (int i = 0; i < d_in; i++) {
            v_out[i] = eigenvectors[i*d_out+idx];
        }
    }

    void save(const char *filename) {
        if (!computed) compute();
        FILE *f = fopen(filename, "wb");
        fwrite(eigenvectors, sizeof(double), d_out*d_in, f);
        fclose(f);
    }

    void compute() {
        // first remove the mean and normalize by the count
        for (int i = 0; i < d_in; i++) {
            for (int j = 0; j < d_in; j++) {
                covariance[i*d_in+j] -= mean[i]*mean[j]/count;
                covariance[i*d_in+j] /= count;
            }
        }        

        // now compute the eigenvectors
        // TODO: do this using a non-retarded algorithm
        for (int i = 0; i < d_in; i++) {
            for (int j = 0; j < d_out; j++) {
                eigenvectors[i*d_out+j] = covariance[i*d_in+j];
            }
        }
        while (1) {
            // orthonormalize
            for (int i = 0; i < d_out; i++) {
                // first make this column independent of all the
                // previous columns                
                for (int j = 0; j < i; j++) {
                    // compute the dot product
                    double dot = 0;
                    for (int k = 0; k < d_in; k++) {
                        dot += eigenvectors[k*d_out+i]*eigenvectors[k*d_out+j];
                    }
                    // The previous column is of unit length, so it's
                    // easy to make this one independent
                    for (int k = 0; k < d_in; k++) {
                        eigenvectors[k*d_out+i] -= eigenvectors[k*d_out+j]*dot;
                    }
                }

                // now normalize this column
                double dot = 0;
                for (int k = 0; k < d_in; k++) {
                    dot += eigenvectors[k*d_out+i]*eigenvectors[k*d_out+i];
                }
                dot = ::sqrt(dot);

                dot = 1.0/dot;

                // make sure the first element of each eigenvector is positive
                if (eigenvectors[i]*dot < 0) dot = -dot;

                for (int k = 0; k < d_in; k++) {
                    eigenvectors[k*d_out+i] *= dot;
                }

            }

            /*
            printf("eigenvector matrix:\n");
            for (int i = 0; i < d_in; i++) {
                for (int j = 0; j < d_out; j++) {
                    printf("%3.4f ", eigenvectors[i*d_out+j]);
                }
                printf("\n");
            }            
            */

            // check for convergence
            double dist = 0;
            for (int i = 0; i < d_in; i++) {
                for (int j = 0; j < d_out; j++) {
                    double delta = tmp[i*d_out+j] - eigenvectors[i*d_out+j];
                    dist += delta*delta;
                }
            }
            if (dist < 0.00001) break;
            
            // multiply by the covariance matrix
            for (int i = 0; i < d_in; i++) {
                for (int j = 0; j < d_out; j++) {
                    tmp[i*d_out+j] = 0;
                    for (int k = 0; k < d_in; k++) {
                        tmp[i*d_out+j] += covariance[i*d_in+k]*eigenvectors[k*d_out+j];
                    }
                }
            }
            double *t = tmp;
            tmp = eigenvectors;
            eigenvectors = t;

            
        }

        computed = true;
    }

  private:

    int d_in, d_out;
    double *covariance, *mean, *eigenvectors, *tmp;
    bool computed;
    int count;
};

#include "footer.h"
#endif
