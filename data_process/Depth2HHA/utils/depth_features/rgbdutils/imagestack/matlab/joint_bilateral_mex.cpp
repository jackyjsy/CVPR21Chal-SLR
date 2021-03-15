// AUTORIGHTS
// ---------------------------------------------------------
// Copyright (c) 2014, Saurabh Gupta
// 
// This file is part of the RGBD Utils code and is available 
// under the terms of the Simplified BSD License provided in 
// LICENSE. Please retain this notice and LICENSE if you use 
// this file (or any portion of it) in your project.
// ---------------------------------------------------------


#include <string>
#include <vector>

#include "mex.h"

#include "ImageStack.h"

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

using namespace ImageStack;  // NOLINT(build/namespaces)

Image fromMat(const mxArray *mxfeat){
  //int frames, int width, int height, int channels) {
  const int *dims = mxGetDimensions(mxfeat);
  if (mxGetNumberOfDimensions(mxfeat) != 4 || mxGetClassID(mxfeat) != mxSINGLE_CLASS)
    mexErrMsgTxt("Invalid input");
  
  int frames = dims[0];
  int channels = dims[1];
  int width = dims[2];
  int height = dims[3];

  int size = width * height * channels * frames;
  float *feat = (float *)mxGetPr(mxfeat);

  Image im(width, height, frames, channels);

  float *dstPtr = im(0, 0, 0);
  float *srcPtr = feat;
  for (int i = 0; i < size; i++) {
    *dstPtr++ = (float)(*srcPtr++);
  }
  return im;
}


mxArray* toMat(Window im) { 
  int size = im.width * im.height * im.channels * im.frames;
  int out[4];
  out[0] = im.frames;
  out[1] = im.channels;
  out[2] = im.width;
  out[3] = im.height;
  mxArray *mxfeat = mxCreateNumericArray(4, out, mxSINGLE_CLASS, mxREAL);
  float *feat = (float *)mxGetPr(mxfeat);
  
  float *srcPtr = im(0, 0, 0);
  for (int i = 0; i < size; i++) {
    *feat++ = (float)(*srcPtr++);
  }
  return mxfeat;
}

void mexFunction(MEX_ARGS) {
  if (nrhs != 4) {
    mexErrMsgTxt("Expecting 4 arguments");
    return;
  }
  if (nlhs > 1){
    mexErrMsgTxt("Max outputs is 1");
    return;
  }
  Image im = fromMat(prhs[0]);
  Image ref = fromMat(prhs[1]);
  float colorSigma = (float) mxGetScalar(prhs[2]);
  float filterHeight = (float) mxGetScalar(prhs[3]);
  float filterWidth = filterHeight;

  // void JointBilateral::apply(Window im, Window ref, 
  //   float filterWidth, float filterHeight, float filterFrames, float colorSigma, 
  //   GaussTransform::Method method);
  // Image im(200, 200, 3, 3); //im.width, im.height, im.frames, im.channels);
  // Image ref(200, 200, 3, 3); //im.width, im.height, im.frames, im.channels);
  
  GaussTransform::Method m = GaussTransform::AUTO;
  JointBilateral::apply(im, ref, filterWidth, filterHeight, 0, colorSigma, m);
  plhs[0] = toMat(im);
}
