#include "main.h"
#include "WLS.h"
#include "Calculus.h"
#include "Color.h"
#include "Arithmetic.h"
#include "Convolve.h"
#include "Geometry.h"
#include "Paint.h"
#include "Statistics.h"
#include "LAHBPCG.h"
#include "header.h"

void WLS::help() {
    pprintf("-wls filters the image with the wls-filter described in the paper"
            " Edge-Preserving Decompositions for Multi-Scale Tone and Detail"
            " Manipulation by Farbman et al. The first parameter (alpha) controls"
            " the sensitivity to edges, and the second one (lambda) controls the"
            " amount of smoothing.\n"
            "\n"
            "Usage: ImageStack -load in.jpg -wls 1.2 0.25 -save blurry.jpg\n");            
}


void WLS::parse(vector<string> args) {
    float alpha = 0, lambda = 0;

    assert(args.size() == 2, "-wls takes two arguments");

    alpha = readFloat(args[0]);
    lambda = readFloat(args[1]);

    Image im = apply(stack(0), alpha, lambda, 0.01);
    
    pop();
    push(im);
}

Image WLS::apply(Window im, float alpha, float lambda, float tolerance) {
        
    Image L;

    // Precalculate the log-luminance differences Lx and Ly
    if(im.channels == 3) {
        L = ColorConvert::apply(im, "rgb", "y");
    } else {
        vector<float> mat;
        for (int c = 0; c < im.channels; c++) {
            mat.push_back(1.0f/im.channels);
        }
        L = ColorMatrix::apply(im, mat);
    }
    
    Stats s(L);
    // If min(s) is less than zero, chanses are that we already are in the log-domain. 
    // In any case, we cannot take the log of negative numbers..
    if(s.minimum() >= 0) {
        Offset::apply(L, 0.0001);
        Log::apply(L);        
    }
    
    Image Lx = L.copy();
    Gradient::apply(Lx, 'x');
    
    Image Ly = L.copy();
    Gradient::apply(Ly, 'y');
    
    // Lx = lambda / (|dl(p)/dx|^alpha + eps)
    // Ly = lambda / (|dl(p)/dy|^alpha + eps)
    for (int t = 0; t < L.frames; t++) {
        for (int y = 0; y < L.height; y++) {
            for (int x = 0; x < L.width; x++) {
                Lx(x, y, t)[0] = lambda / (powf(fabs(Lx(x, y, t)[0]), alpha) + 0.0001);
                Ly(x, y, t)[0] = lambda / (powf(fabs(Ly(x, y, t)[0]), alpha) + 0.0001);
            }
            // Nuke the weights for the boundary condition
            Lx(0, y, t)[0] = 0;
        }
        for (int x = 0; x < L.width; x++) {
            Ly(x, 0, t)[0] = 0;
        }
    }
    
    // Data weights equal to 1 all over...
    Image w(im.width, im.height, 1, 1);
    Offset::apply(w, 1); 
    
    // For this filter gx and gy is 0 all over (target gradient is smooth)
    Image zeros(im.width, im.height, 1, im.channels);
    
    // Solve using the fast preconditioned conjugate gradient.
    Image x = LAHBPCG::apply(im, zeros, zeros, w, Lx, Ly, 200, tolerance);
    
    return x;
}
#include "footer.h"
