#ifndef IMAGESTACK_LINEARALGEBRA_H
#define IMAGESTACK_LINEARALGEBRA_H
#include <stdio.h>
#include <math.h>
#include "header.h"

/* 

This class takes a set of vectors of length N and vectors of length M,
and computes the best (least-squares) matrix that maps from the first
set to the second. Ie, given a bunch of examples, it solves Ax = b for
A. Written this way, it returns A in *column-major* format.

You construct a solver that maps from N dimensional vectors to M
dimensional vectors like so:

LeastSquaresSolver<N, M> solver;

One of the two methods is:

void solver.addCorrespondence(float *, float *);

which takes two example vectors and updated some internal state. You
may also pass a weight as an optional third argument. There is also a
double version of this method, so floats aren't necessary.

The other method is:

bool solver.solve(double *);

which fills in the argument with the best matrix in column major
format. It will return false if not enough examples have been
given (or if they are linearly dependent), otherwise it returns true.

It operates efficiently in space, using O(N*N)+O(M*N) memory,
regardless of how many examples are given, so feel free to go nuts
with examples. It is also efficient in time, using a direct Cholesky
decomposition to do the final solve.

Example Usage:

Say we wish to compute an affine warp from some 2D set of vectors x to
some other 2D set of vectors y.

-----------------------------------------------------------
LeastSquaresSolver<3, 2> solver;

// these could also be doubles
float x[3], y[2];

while (there are more examples) {
   // do something to fill in x[0], x[1], y[0], y[1] with the next example input and output
   
   // set x[2] to 1, because we're computing an affine transformation using a matrix
   x[2] = 1;

   solver.addCorrespondence(x, y);
}

double matrix[2*3];

if (!solver.solve(matrix)) {
  // underconstrained
}
// matrix now contains the best affine transform in column major order
-----------------------------------------------------------

Say we wish to instead compute a polynomial map from x to y, where y can
include quadratic terms in x:

-----------------------------------------------------------
LeastSquaresSolver<6, 2> solver;

float x[6], y[2];

while (there are more examples) {
   // do something to fill in x[0], x[1], y[0], y[1] with the next example
   
   // fill in the rest of the quadratic terms
   x[2] = 1;
   x[3] = x[0]*x[0];
   x[4] = x[0]*x[1];
   x[5] = x[1]*x[1];

   solver.addCorrespondence(x, y);
}

double m[2*6];

if (!solver.solve(m)) {
   // underconstrained!
}

// m now contains the best quadratic transform in column major order
// ie it contains the values that best make these equations true:

// y0 = m0*x0 + m2*x1 + m4*x2 + m6*x3 + m8*x4 + m10*x5
// y1 = m1*x0 + m3*x1 + m5*x2 + m7*x3 + m9*x4 + m11*x5

// or, equivalently:

// y0 = m0*x0 + m2*x1 + m4 + m6*x0*x0 + m8*x0*x1 + m10*x1*x1
// y1 = m1*x0 + m3*x1 + m5 + m7*x0*x0 + m9*x0*x1 + m11*x1*x1

-----------------------------------------------------------


*/

template<int N, int M>
class LeastSquaresSolver {
public:    
    // stored row major
    double AtA[N*N];
    double Atb[N*M];

    LeastSquaresSolver() {
        reset();
    }

    void reset() {
        for (int i = 0; i < N*N; i++) {
            AtA[i] = 0;
        }
        for (int i = 0; i < N*M; i++) {
            Atb[i] = 0;
        }
    }

    void addCorrespondence(float *in, float *out, float weight) {
        // update AtA
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                AtA[i*N+j] += in[i]*in[j]*weight;
            }
        }        

        // update Atb
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                Atb[i*M+j] += in[i]*out[j]*weight;
            }
        }        
    }

    void addCorrespondence(float *in, float *out) {
        // update AtA
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                AtA[i*N+j] += in[i]*in[j];
            }
        }        

        // update Atb
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                Atb[i*M+j] += in[i]*out[j];
            }
        }        
    }

    void addCorrespondence(double *in, double *out, double weight) {
        // update AtA
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                AtA[i*N+j] += in[i]*in[j]*weight;
            }
        }        

        // update Atb
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                Atb[i*M+j] += in[i]*out[j]*weight;
            }
        }        
    }

    void addCorrespondence(double *in, double *out) {
        // update AtA
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                AtA[i*N+j] += in[i]*in[j];
            }
        }        

        // update Atb
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                Atb[i*M+j] += in[i]*out[j];
            }
        }        
    }

    // returns whether or not it succeeded
    bool solve(double *solution) {
        bool isspd = true;

        double L[N*N];
        for (int i = 0; i < N*N; i++) {
            L[i] = 0.0;
        }
        // compute cholesky decomposition A = L'*L
        for (int j = 0; j < N; j++) {
            double d = 0.0;
            for (int k = 0; k < j; k++) {
                double s = 0.0;
                for (int i = 0; i < k; i++) {
                    s += L[k*N+i]*L[j*N+i];
                }
                L[j*N+k] = s = (AtA[j*N+k] - s)/L[k*N+k];
                d = d + s*s;
                isspd = isspd && (AtA[k*N+j] == AtA[j*N+k]); 
            }
            d = AtA[j*N+j] - d;
            isspd = isspd && (d > 0.0);
            L[j*N+j] = sqrt(d > 0.0 ? d : 0.0);
            for (int k = j+1; k < N; k++) {
                L[j*N+k] = 0.0;
            }
        }                

        // bail if not symmetric positive definite
        if (!isspd) return false;

        for (int i = 0; i < M*N; i++) solution[i] = Atb[i];

        // apply the decomposition to produce a solution
        // Solve L*y = b;
        for (int j = 0; j < M; j++) {
            for (int k = 0; k < N; k++) {
                for (int i = 0; i < k; i++) 
                    solution[j+k*M] -= solution[j+i*M]*L[k*N+i];
                solution[j+k*M] /= L[k*N+k];            
            }

            // Solve L'*X = Y;
            for (int k = N-1; k >= 0; k--) {
                for (int i = k+1; i < N; i++) 
                    solution[j+k*M] -= solution[j+i*M]*L[i*N+k];
                solution[j+k*M] /= L[k*N+k];
            }
        }


        return true;
    }    
};

#include "footer.h"
#endif
