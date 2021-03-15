## RGBD Utils

Utilities for handling depth images. Provides functions for computing normals, estimating direction of gravity, and normal and depth gradients (as designed in our CVPR 2013 paper) from depth images from a Kinect.


### Installation

0. Compile ImageStack
We use the ImageStack library for joint bilateral filtering operations (https://code.google.com/p/imagestack/). To compile the library.
```
cd imagestack
make all
cd ..
```

### Citing

If you find this code useful in your research, please consider citing:

    @incollection{guptaCVPR13,
      author = {Saurabh Gupta and Pablo Arbelaez and Jitendra Malik},
      title = {Perceptual Organization and Recognition of Indoor Scenes from {RGB-D} Images},
      booktitle = {CVPR},
      year = {2013},
    }

### License

RGBD Utils are released under the Simplified BSD License (refer to the LICENSE file for details).
