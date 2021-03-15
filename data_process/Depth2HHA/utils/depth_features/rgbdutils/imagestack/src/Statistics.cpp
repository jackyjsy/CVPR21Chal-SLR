#include "main.h"
#include "Statistics.h"
#include "Calculus.h"
#include "Arithmetic.h"
#include "eigenvectors.h"
#include <algorithm>
#include <iostream>
#include "header.h"

void Dimensions::help() {
    pprintf("-dimensions prints the size of the current image.\n\n"
           "Usage: ImageStack -load a.tga -dimensions\n");
}

void Dimensions::parse(vector<string> args) {
    assert(args.size() == 0, "-dimensions takes no arguments\n");
    printf("Width x Height x Frames x Channels: %d x %d x %d x %d\n",
           stack(0).width, stack(0).height, stack(0).frames, stack(0).channels);
}

Stats::Stats(Window im) : im_(im) {
    sum_ = mean_ = variance_ = skew_ = kurtosis_ = 0;

    channels = im.channels;

    min_ = max_ = im.data[0];

    for (int c = 0; c < im.channels; c++) {
        means.push_back(0);
        sums.push_back(0);
        variances.push_back(0);
        kurtoses.push_back(0);
        skews.push_back(0);
        mins.push_back(im.data[c]);
        maxs.push_back(im.data[c]);
        spatialvariances.push_back(0);
        spatialvariances.push_back(0);
        barycenters.push_back(0);
        barycenters.push_back(0);

        for (int c2 = 0; c2 < im.channels; c2++) {
            covarianceMatrix.push_back(0);
        }
    }

    basicStatsComputed = false;
    momentsComputed = false;
}

void Stats::computeBasicStats() {
    vector<int> counts(im_.channels, 0);
    nans_ = posinfs_ = neginfs_ = 0;
    int count = 0;
    for (int t = 0; t < im_.frames; t++) {
        for (int y = 0; y < im_.height; y++) {
            for (int x = 0; x < im_.width; x++) {
                for (int c = 0; c < im_.channels; c++) {
                    float val = im_(x, y, t)[c];
                    if (!isfinite(val)) {
                        if (isnan(val)) nans_++;
                        else if (val > 0) posinfs_++;
                        else neginfs_++;
                        continue;
                    };
                    counts[c]++;
                    count++;
                    sum_ += val;
                    sums[c] += val;
                    // printf("%.5lf |",val);
                    if (val < min_) min_ = val;
                    if (val < mins[c]) mins[c] = val;
                    if (val > max_) max_ = val;
                    if (val > maxs[c]) maxs[c] = val;
                }
            }
        }
    }

    mean_ = sum_ / count;
    for (int c = 0; c < im_.channels; c++) {
        means[c] = sums[c] / counts[c];
    }

    basicStatsComputed = true;
}

void Stats::computeMoments() {
    // figure out variance, skew, and kurtosis
    vector<int> counts(im_.channels, 0);
    int count = 0;
    vector<int> covarianceCounts(channels * channels, 0);

    for (int t = 0; t < im_.frames; t++) {
        for (int y = 0; y < im_.height; y++) {
            for (int x = 0; x < im_.width; x++) {
                for (int c = 0; c < im_.channels; c++) {
                    float val = im_(x, y, t)[c];
                    if (!isfinite(val)) continue;
                    counts[c]++;
                    count++;
                    float diff = (float)(val - means[c]);
                    for (int c2 = 0; c2 < im_.channels; c2++) {
                        float val2 = im_(x, y, t)[c2];
                        if (!isfinite(val2)) continue;
                        float diff2 = (float)(val2 - means[c2]);
                        covarianceMatrix[c * channels + c2] += diff * diff2;
                        covarianceCounts[c * channels + c2]++;
                    }
                    float power = diff * diff;
                    barycenters[c*2] += x*val;
                    barycenters[c*2+1] += y*val;
                    spatialvariances[c*2] += x*x*val;
                    spatialvariances[c*2+1] += y*y*val;
                    variances[c] += power;
                    variance_ += power;
                    power *= diff;
                    skews[c] += power;
                    skew_ += power;
                    power *= diff;
                    kurtosis_ += power * diff;
                    kurtoses[c] += power * diff;
                }
            }
        }
    }

    variance_ /= (count - 1);
        skew_ /= (count - 1) * variance_ * ::sqrt(variance_);
    kurtosis_ /= (count - 1) * variance_ * variance_;
    for (int c = 0; c < im_.channels; c++) {
        for (int c2 = 0; c2 < im_.channels; c2++) {
            covarianceMatrix[c * channels + c2] /= covarianceCounts[c * channels + c2] - 1;
        }
        variances[c] /= (counts[c] - 1);
                skews[c] /= (counts[c] - 1) * variances[c] * ::sqrt(variances[c]);
        kurtoses[c] /= (counts[c] - 1) * variances[c] * variances[c];
    }
    
    for (int c = 0; c < im_.channels; c++) {
        barycenters[c*2] /= sums[c];
        barycenters[c*2+1] /= sums[c];
        spatialvariances[c*2] /= sums[c];
        spatialvariances[c*2] -= barycenters[c*2] * barycenters[c*2];
        spatialvariances[c*2+1] /= sums[c];
        spatialvariances[c*2+1] -= barycenters[c*2+1] * barycenters[c*2+1];        
    }
    momentsComputed = true;
}


void Statistics::help() {
    pprintf("-statistics provides per channel statistical information about the current image.\n\n"
            "Usage: ImageStack -load a.tga -statistics\n");
}

void Statistics::parse(vector<string> args) {
    assert(args.size() == 0, "-statistics takes no arguments");

    apply(stack(0));
}

void Statistics::apply(Window im) {
    Stats stats(im);

    printf("Frames x Width x Height x Channels: %i %i %i %i\n", im.frames, im.width, im.height, im.channels);

    printf("Minima:  \t\t");
    for (int i = 0; i < im.channels; i++) {
        printf("%3.6f\t", stats.minimum(i));
    }        
    printf("\n");

    printf("Maxima:  \t\t");
    for (int i = 0; i < im.channels; i++) {
        printf("%3.6f\t", stats.maximum(i));
    }
    printf("\n");

    printf("Sums:    \t\t");
    for (int i = 0; i < im.channels; i++) {
        printf("%3.6f\t", stats.sum(i));
    }
    printf("\n");

    printf("Means:   \t\t");
    for (int i = 0; i < im.channels; i++) {
        printf("%3.6f\t", stats.mean(i));
    }
    printf("\n");

    printf("Variance:\t\t");
    for (int i = 0; i < im.channels; i++) {
        printf("%3.6f\t", stats.variance(i));
    }
    printf("\n");

    printf("Covariance Matrix:\n");
    for (int i = 0; i < im.channels; i++) {
        printf("\t\t\t");
        for (int j = 0; j < im.channels; j++) {
            printf("%3.6f\t", stats.covariance(i, j));
        }
        printf("\n");
    }
    printf("\n");
    
    printf("Skewness:\t\t");
    for (int i = 0; i < im.channels; i++) {
        printf("%3.6f\t", stats.skew(i));
    }
    printf("\n");

    printf("Kurtosis:\t\t");
    for (int i = 0; i < im.channels; i++) {
        printf("%3.6f\t", stats.kurtosis(i));
    }
    printf("\n");
    printf("\n");

    printf("Barycenter (X):\t\t");
    for (int i = 0; i < im.channels; i++) {
      printf("%3.6f\t", stats.barycenterX(i));
    }
    printf("\n");

    printf("Barycenter (Y):\t\t");
    for (int i = 0; i < im.channels; i++) {
      printf("%3.6f\t", stats.barycenterX(i));
    }
    printf("\n");

    printf("Spatial variance (X):\t");
    for (int i = 0; i < im.channels; i++) {
      printf("%3.6f\t", stats.spatialvarianceX(i));
    }
    printf("\n");

    printf("Spatial variance (Y):\t");
    for (int i = 0; i < im.channels; i++) {
      printf("%3.6f\t", stats.spatialvarianceY(i));
    }
    printf("\n");
    printf("\n");

    printf("NaN count: %i\n", stats.nans());
    printf("+Inf count: %i\n", stats.posinfs());
    printf("-Inf count: %i\n", stats.neginfs());
    printf("\n");
}


void Noise::help() {
    pprintf("-noise adds uniform noise to the current image, uncorrelated across the"
            " channels, in the range between the two arguments. With one argument, the"
            " lower value is assumed to be zero. With no arguments, the range is assumed"
            " to be [0, 1]\n\n"
            "Usage: ImageStack -load a.tga -push -noise -add -save anoisy.tga\n\n");
}

void Noise::parse(vector<string> args) {
    assert(args.size() < 3, "-noise takes zero, one, or two arguments\n");

    float maxVal = 1;
    float minVal = 0;
    if (args.size() == 1) {
        maxVal = readFloat(args[0]);
    } else if (args.size() == 2) {
        minVal = readFloat(args[0]);
        maxVal = readFloat(args[1]);
    }

    apply(stack(0), minVal, maxVal);

}

void Noise::apply(Window im, float minVal, float maxVal) {
    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                for (int c = 0; c < im.channels; c++) {
                    im(x, y, t)[c] += randomFloat(minVal, maxVal);
                }
            }
        }
    }
}


void Histogram::help() {
    pprintf("-histogram computes a per-channel histogram of the current image."
            " The first optional argument specifies the number of buckets in the"
            " histogram. If this is not given it defaults to 256. The second and"
            " third arguments indicate the range of data to expect. These default"
            " to 0 and 1."
            "Usage: ImageStack -load a.tga -histogram -normalize -plot 256 256 3 -display\n");
}


void Histogram::parse(vector<string> args) {
    assert(args.size() < 4, "-histogram takes three or fewer arguments\n");
    int buckets = 256;
    float minVal = 0.0f;
    float maxVal = 1.0f;
    if (args.size() > 0) {
        buckets = readInt(args[0]);
    }
    if (args.size() > 1) {
        minVal = readFloat(args[1]);
    }
    if (args.size() > 2) {
        maxVal = readFloat(args[2]);
    }

    push(apply(stack(0), buckets, minVal, maxVal));
}


Image Histogram::apply(Window im, int buckets, float minVal, float maxVal) {

    float invBucketWidth = buckets / (maxVal - minVal);

    float inc = 1.0f / (im.width * im.height * im.frames);

    Image hg(buckets, 1, 1, im.channels);

    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                for (int c = 0; c < im.channels; c++) {
                    double value = im(x, y, t)[c];
                    int bucket;
                    if (isnan((float)value)) {
                        continue;
                    } else if (isinf((float)value)) {
                        continue;
                    } else {
                        bucket = (int)((value - minVal) * invBucketWidth);
                        if (bucket >= buckets) bucket = buckets-1;
                        if (bucket < 0) bucket = 0;
                    }
                    hg(bucket, 0, 0)[c] += inc;
                }
            }
        }
    }

    return hg;
}



void Equalize::help() {
    pprintf("-equalize flattens out the histogram of an image, while preserving ordering"
            " between pixel brightnesses. It does this independently in each channel. When"
            " given no arguments, it produces an image with values between zero and one. With"
            " one argument, it produces values between zero and that argument. With two"
            " arguments, it produces values between the two arguments. The brightest pixel(s)"
            " will always map to the upper bound and the dimmest to the lower bound.\n\n"
            "Usage: ImageStack -load a.tga -equalize 0.2 0.8 -save out.tga\n\n");
}

void Equalize::parse(vector<string> args) {
    assert(args.size() < 3, "-equalize takes zero, one, or two arguments\n");
    float lower = 0, upper = 1;
    if (args.size() == 1) upper = readFloat(args[0]);
    else if (args.size() == 2) {
        lower = readFloat(args[0]);
        upper = readFloat(args[1]);
    }
    apply(stack(0), lower, upper);
}

void Equalize::apply(Window im, float lower, float upper) {
    Stats stats(im);

    // STEP 1) Normalize the image to the 0-1 range
    Normalize::apply(im);

    // STEP 2) Calculate a CDF of the image
    int buckets = 4096;
    Image cdf = Histogram::apply(im, buckets);
    Integrate::apply(cdf, 'x');

    // STEP 3) For each pixel, find out how many things are in the same bucket or smaller, and use that to set the value
    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                for (int c = 0; c < im.channels; c++) {
                    float alpha = im(x, y, t)[c] * buckets;
                    int bucket = (int)alpha; // which bucket am I in?
                    if (bucket < 0) bucket = 0;
                    if (bucket >= buckets) bucket = buckets-1;
                    alpha -= bucket; // how far along am I in this bucket?
                    
                    // how many pixels are probably less than me?
                    float lesser = 0;
                    if (bucket > 0) lesser = cdf(bucket-1, 0)[c];
                    float equal = cdf(bucket, 0)[c] - lesser;

                    // use the estimate to set the value
                    im(x, y, t)[c] = (lesser + alpha * equal) * (upper - lower) + lower;
                }
            }
        }
    } 
}




void HistogramMatch::help() {
    pprintf("-histogrammatch alters the histogram of the current image to match"
            " that of the second image, while preserving ordering. Performing any"
            " monotonic operation to an image, and then histogram matching it to"
            " its original should revert it to its original.\n\n"
            "Usage: ImageStack -load a.tga -load b.tga -histogrammatch -save ba.tga\n\n");
}

void HistogramMatch::parse(vector<string> args) {
    assert(args.size() == 0, "-histogrammatch takes no arguments\n");
    apply(stack(0), stack(1));
}

void HistogramMatch::apply(Window im, Window model) {
    assert(im.channels == model.channels, "Images must have the same number of channels\n");

    // Compute cdfs of the two images
    Stats s1(im), s2(model);
    int buckets = 4096;
    Image cdf1 = Histogram::apply(im, buckets, s1.minimum(), s1.maximum());
    Image cdf2 = Histogram::apply(model, buckets, s2.minimum(), s2.maximum()); 
    Integrate::apply(cdf1, 'x');
    Integrate::apply(cdf2, 'x');

    // Invert cdf2
    Image inverseCDF2(cdf2.width, 1, 1, cdf2.channels);
    for (int c = 0; c < inverseCDF2.channels; c++) {
        int xi = 0;
        float invWidth = 1.0f / cdf2.width;
        for (int x = 0; x < inverseCDF2.width; x++) {            
            while (cdf2(xi, 0)[c] < x * invWidth && xi < cdf2.width) xi++;
            // cdf2(xi, 0)[c] is now just greater than x / inverseCDF2.width
            float lower = xi > 0 ? cdf2(xi-1, 0)[c] : 0;
            float upper = xi < cdf2.width ? cdf2(xi, 0)[c] : lower;

            // where is x*invWidth between lower and upper?
            float alpha = 0;
            if (upper > lower && x*invWidth >= lower && x*invWidth <= upper)
                alpha = (x*invWidth - lower)/(upper - lower);
            
            inverseCDF2(x, 0)[c] = xi + alpha;            
        }
    }

    // Now apply the cdf of image 1 followed by the inverse cdf of image 2
    float invBucketWidth = buckets / (s1.maximum() - s1.minimum());
    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                for (int c = 0; c < im.channels; c++) {
                    float alpha = (im(x, y, t)[c] - s1.minimum()) * invBucketWidth;
                    int bucket = (int)alpha; // which bucket am I in?
                    if (bucket < 0) bucket = 0;
                    if (bucket >= buckets) bucket = buckets-1;
                    alpha -= bucket; // how far along am I in this bucket?
                    
                    // how many pixels are probably less than me?
                    float lesser = 0;
                    if (bucket > 0) lesser = cdf1(bucket-1, 0)[c];
                    float equal = cdf1(bucket, 0)[c] - lesser;

                    // use the estimate to get the percentile
                    float percentile = lesser + alpha * equal;

                    //im(x, y, t)[c] = percentile;

                    
                    // look up the percentile in inverseCDF2 in the same way
                    alpha = percentile * buckets;
                    bucket = (int)alpha;
                    if (bucket < 0) bucket = 0;
                    if (bucket >= buckets) bucket = buckets-1;
                    alpha -= bucket;
                        
                    lesser = 0;
                    if (bucket > 0) lesser = inverseCDF2(bucket-1, 0)[c];
                    equal = inverseCDF2(bucket, 0)[c] - lesser;
                    
                    im(x, y, t)[c] = (lesser + alpha * equal) * (s2.maximum() - s2.minimum()) / buckets + s2.minimum();
                    
                }
            }
        }
    }     
}


void Shuffle::help() {
    pprintf("-shuffle takes every pixel in the current image and swaps it to a"
            " random new location creating a new noise image with exactly the same"
            " histogram.\n\n"
            "Usage: ImageStack -load a.tga -shuffle -save shuffled.tga\n\n");
}

void Shuffle::parse(vector<string> args) {
    assert(args.size() == 0, "-shuffle takes no arguments\n");
    apply(stack(0));
}

void Shuffle::apply(Window im) {
    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                float *ptr1 = im(x, y, t);
                // pick a random new location after this one
                int ot, ox, oy;
                ot = randomInt(t, im.frames-1);
                if (ot > t)
                    oy = randomInt(0, im.height-1);
                else 
                    oy = randomInt(y, im.height-1);
                if (oy > y || ot > t) 
                    ox = randomInt(0, im.width-1);
                else
                    ox = randomInt(x+1, im.width-1);

                float *ptr2 = im(ox, oy, ot);
                for (int c = 0; c < im.channels; c++) {
                    float tmp = ptr1[c];
                    ptr1[c] = ptr2[c];
                    ptr2[c] = tmp;
                }
            }
        }
    }
}

void KMeans::help() {
    printf("-kmeans clusters the image into a number of clusters given by the"
           " single integer argument.\n\n"
           "Usage: ImageStack -load in.jpg -kmeans 3 -save out.jpg\n\n");
}

void KMeans::parse(vector<string> args) {
    assert(args.size() == 1, "-kmeans takes one argument\n");
    apply(stack(0), readInt(args[0]));
}

void KMeans::apply(Window im, int clusters) {
    assert(clusters > 1, "must have at least one cluster\n");

    vector< vector<float> > cluster, newCluster;
    vector<int> newClusterMembers(clusters);

    for (int c = 0; c < im.channels; c++) {
        cluster.push_back(vector<float>(clusters, 0));
        newCluster.push_back(vector<float>(clusters, 0));
    }

    // initialize the clusters to randomly selected pixels
    for (int i = 0; i < clusters; i++) {
        float *pixel = im(randomInt(0, im.width-1), randomInt(0, im.height-1), randomInt(0, im.frames-1));
        for (int c = 0; c < im.channels; c++) {
            cluster[c][i] = pixel[c] + randomFloat(-0.0001f, 0.0001f);
        }
    }

    while (1) {
        // wipe the newCluster
        for (int i = 0; i < clusters; i++) {
            newClusterMembers[i] = 0;
            for (int c = 0; c < im.channels; c++) {           
                newCluster[c][i] = 0;
            }
        }

        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {
                for (int x = 0; x < im.width; x++) {
                    float *pixel = im(x, y, t);
                    // assign this pixel to a cluster
                    int bestCluster = 0;
                    float bestDistance = 1e10;
                    for (int i = 0; i < clusters; i++) {
                        float distance = 0;
                        for (int c = 0; c < im.channels; c++) {
                            float d = cluster[c][i] - pixel[c];
                            distance += d*d;
                        }
                        if (distance < bestDistance) {
                            bestCluster = i;
                            bestDistance = distance;
                        }
                    }

                    for (int c = 0; c < im.channels; c++) {
                        newCluster[c][bestCluster] += pixel[c];
                    }
                    newClusterMembers[bestCluster]++;
                }
            }
        }

        // normalize the new clusters (reset any zero ones to random)
        for (int i = 0; i < clusters; i++) {
            if (newClusterMembers[i] == 0) {
                float *pixel = im(randomInt(0, im.width-1),
                                         randomInt(0, im.height-1),
                                         randomInt(0, im.frames-1));
                for (int c = 0; c < im.channels; c++) {
                    newCluster[c][i] = pixel[c] + randomFloat(-0.0001f, 0.0001f);
                }
            } else {
                for (int c = 0; c < im.channels; c++) {
                    newCluster[c][i] /= newClusterMembers[i];
                }
            }
        }

        // break if the clusters are unchanged
        if (cluster == newCluster) break;

        // swap over the pointers
        cluster.swap(newCluster);
    }

    // now color each pixel according to the closest cluster
    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                float *pixel = im(x, y, t);
                // assign this pixel to a cluster
                int bestCluster = 0;
                float bestDistance = 1e10;
                for (int i = 0; i < clusters; i++) {
                    float distance = 0;
                    for (int c = 0; c < im.channels; c++) {
                        float d = cluster[c][i] - pixel[c];
                        distance += d*d;
                    }
                    if (distance < bestDistance) {
                        bestCluster = i;
                        bestDistance = distance;
                    }
                }

                for (int c = 0; c < im.channels; c++) {
                    pixel[c] = cluster[c][bestCluster];
                }
            }
        }
    }    
}

void Sort::help() {
    pprintf("-sort sorts the data along the given dimension for every value of the"
            " other dimensions. For example, the following command computes the"
            " median frame of a video.\n\n"
            "ImageStack -loadframes frame*.jpg -sort t -crop frames/2 1 -save median.jpg\n\n");
}

void Sort::parse(vector<string> args) {
    assert(args.size() == 1, "-sort takes one argument\n");
    apply(stack(0), readChar(args[0]));
}

void Sort::apply(Window im, char dimension) {
    assert(dimension == 'x' || dimension == 'y' || dimension == 't' || dimension == 'c',
           "Dimension must be x, y, t, or c\n");

    if (dimension == 'c') {
        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {
                for (int x = 0; x < im.width; x++) {
                                        ::std::sort(im(x, y, t), im(x, y, t)+im.channels);
                }
            }
        }
    } else if (dimension == 'x') {
        vector<float> tmp(im.width);
        for (int t = 0; t < im.frames; t++) {
            for (int y = 0; y < im.height; y++) {
                for (int c = 0; c < im.channels; c++) {
                    for (int x = 0; x < im.width; x++) {
                        tmp[x] = im(x, y, t)[c];
                    }
                    sort(tmp.begin(), tmp.end());
                    for (int x = 0; x < im.width; x++) {
                        im(x, y, t)[c] = tmp[x];
                    }
                }
            }
        }
    } else if (dimension == 'y') {
        vector<float> tmp(im.height);
        for (int t = 0; t < im.frames; t++) {
            for (int x = 0; x < im.width; x++) {
                for (int c = 0; c < im.channels; c++) {
                    for (int y = 0; y < im.height; y++) {                        
                        tmp[y] = im(x, y, t)[c];
                    }
                    sort(tmp.begin(), tmp.end());
                    for (int y = 0; y < im.height; y++) {
                        im(x, y, t)[c] = tmp[y];
                    }
                }
            }
        }

    } else if (dimension == 't') {
        vector<float> tmp(im.frames);
        for (int y = 0; y < im.height; y++) {                        
            for (int x = 0; x < im.width; x++) {                                
                for (int c = 0; c < im.channels; c++) {
                    for (int t = 0; t < im.frames; t++) {                        
                        tmp[t] = im(x, y, t)[c];
                    }
                    sort(tmp.begin(), tmp.end());
                    for (int t = 0; t < im.frames; t++) {
                        im(x, y, t)[c] = tmp[t];
                    }
                }
            }
        }
    }
}



void DimensionReduction::help() {
    pprintf("-dimensionreduction takes a dimensionality and projects all points on"
            " the image onto a linear subspace of best fit with that number of"
            " dimensions. It is useful if you know an image should be low"
            " dimensional (eg a sunset is mostly shades or red), and components"
            " orthogonal to that dimension are unwanted (eg chromatic"
            " abberation).\n\n"
            "Usage: ImageStack -load sunset.jpg -dimensionreduction 2 -save fixed.jpg\n\n");
}

void DimensionReduction::parse(vector<string> args) {
    assert(args.size() == 1, "-dimensionreduction takes no argument\n");
    apply(stack(0), readInt(args[0]));
}

void DimensionReduction::apply(Window im, int dimensions) {
    assert(dimensions < im.channels && dimensions > 0, 
           "dimensions must be greater than zero and less than the current number of channels\n");

    // get some statistics (mostly for the covariance matrix)
    Stats stats(im);

    // we're going to find the leading eigenvectors of the covariance matrix and
    // use them as the basis for our subspace
    vector<float> subspace(dimensions * im.channels), newsubspace(dimensions * im.channels);

    for (int i = 0; i < dimensions * im.channels; i++) subspace[i] = randomFloat(0, 1);

    float delta = 1;
    while (delta > 0.00001) {
        // multiply the subspace by the covariance matrix
        for (int d = 0; d < dimensions; d++) {
            for (int c1 = 0; c1 < im.channels; c1++) {
                newsubspace[d * im.channels + c1] = 0;
                for (int c2 = 0; c2 < im.channels; c2++) {
                    newsubspace[d * im.channels + c1] += (float)(subspace[d * im.channels + c2] * stats.covariance(c1, c2));
                }
            }
        }

        // orthonormalize it with gram-schmidt
        for (int d = 0; d < dimensions; d++) {
            // first subtract it's component in the direction of all earlier vectors
            for (int d2 = 0; d2 < d; d2++) {
                float dot = 0;
                for (int c = 0; c < im.channels; c++) {
                    dot += newsubspace[d * im.channels + c] * newsubspace[d2 * im.channels + c];
                }
                for (int c = 0; c < im.channels; c++) {
                    newsubspace[d * im.channels + c] -= dot * newsubspace[d2 * im.channels + c];
                }                
            }
            // then normalize it
            float sum = 0;
            for (int c = 0; c < im.channels; c++) {
                float val = newsubspace[d * im.channels + c];
                sum += val * val;
            }
            float factor = 1.0f / sqrt(sum);
            for (int c = 0; c < im.channels; c++) {
                newsubspace[d * im.channels + c] *= factor;
            }
            // if the sum is negative, flip it
            sum = 0;
            for (int c = 0; c < im.channels; c++) {
                sum += newsubspace[d * im.channels + c];
            }
            if (sum < -0.01) {
                for (int c = 0; c < im.channels; c++) {
                    newsubspace[d * im.channels + c] *= -1;
                }                
            }
        }

        // check to see how much changed
        delta = 0;
        for (int d = 0; d < dimensions; d++) {
            for (int c = 0; c < im.channels; c++) {
                float diff = newsubspace[d * im.channels + c] - subspace[d * im.channels + c];
                delta += diff * diff;
            }
        }

        newsubspace.swap(subspace);
    }

    // now project the image onto the subspace

    vector<float> output(im.channels);
    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                for (int c = 0; c < im.channels; c++) output[c] = 0;
                // project this pixel onto each vector, and add the results
                for (int d = 0; d < dimensions; d++) {
                    float dot = 0;
                    for (int c = 0; c < im.channels; c++) {
                        dot += im(x, y, t)[c] * subspace[d * im.channels + c];                        
                    }
                    for (int c = 0; c < im.channels; c++) {
                        output[c] += dot * subspace[d * im.channels + c];
                    }
                }
                for (int c = 0; c < im.channels; c++) im(x, y, t)[c] = output[c];
            }
        }
    }

    // display the subspace for debugging
    printf("Basis chosen:\n");
    for (int c = 0; c < im.channels; c++) {
        for (int d = 0; d < dimensions; d++) {
            printf("%f \t", newsubspace[d * im.channels + c]);
        }
        printf("\n");
    }        



}

void LocalMaxima::help() {
    pprintf("-localmaxima finds local maxima in the image and outputs their"
            " locations to a text file. Each line in the text file consists of four"
            " comma-delimited floating point values, corresponding to the t, x and y"
            " coordinates, and the strength of the local maxima (its value minus the"
            " maximum neighbor). -localmaxima will only operate on the first"
            " channel of an image. There are three arguments. The first is some"
            " string containing the characters x, y, and t. It specifies the"
            " dimensions over which a pixel must be greater than its neighbors. The"
            " second is the minimum value by which a pixel must exceed its"
            " neighbors to count as a local maximum. The third is the minimum"
            " distance which must separate adjacent local maxima.\n"
            "\n"
            "Usage: ImageStack -load stack.tmp -localmaxima txy 0.01 5 output.txt\n");
}

void LocalMaxima::parse(vector<string> args) {
    assert(args.size() == 4, "-localmaxima takes 4 arguments\n");
    bool tCheck = false, xCheck = false, yCheck = false;

    // first make sure file can be opened
    FILE *f = fopen(args[3].c_str(), "w");
    assert(f, "Could not open file %s\n", args[2].c_str());    
    
    for(unsigned int i=0; i < args[0].size(); i++) {
        switch(args[0][i]) {
        case 't': 
            tCheck=true;
            break;
        case 'x':
            xCheck=true;
            break;
        case 'y':
            yCheck=true;
            break;
        default:
            panic("Unknown -localmaxima flag: %c\n",args[0][i]);
        }
    }
    assert(tCheck || xCheck || yCheck, "-localmaxima requires at least one active dimension to find local maxima\n");

    vector<LocalMaxima::Maximum> maxima = apply(stack(0), xCheck, yCheck, tCheck, readFloat(args[1]), readFloat(args[2]));
    for (unsigned int i = 0; i < maxima.size(); i++) {
        fprintf(f, "%f,%f,%f,%f\n",
                maxima[i].t,
                maxima[i].x,
                maxima[i].y,
                maxima[i].value);        
    }
   
    fclose(f);
}

struct LocalMaximaCollision {
    // a is the index of stronger of the two
    unsigned a, b;
    
    float disparity;
    
    
    // define an operator so that std::sort will sort them from
    // maximum strength disparity to minimum strength disparity
    bool operator<(const LocalMaximaCollision &other) const {
        return disparity > other.disparity;
    }
};

vector<LocalMaxima::Maximum> LocalMaxima::apply(Window im, bool tCheck, bool xCheck, bool yCheck, float threshold, float minDistance) {

    vector<LocalMaxima::Maximum> results;
    
    // float values for locations
    float ft, fx, fy;
    float value; // the actual image value
    
    // select bounds for search
    int tStart, tEnd, yStart, yEnd, xStart, xEnd;
    if (tCheck) {
        tStart = 1;
        tEnd = im.frames-1;
    } else {
        tStart = 0;
        tEnd = im.frames;
    }
    if (xCheck) {
        xStart = 1;
        xEnd = im.width-1;
    } else {
        xStart = 0;
        xEnd = im.width;
    }
    if (yCheck) {
        yStart = 1;
        yEnd = im.height-1;
    } else {
        yStart = 0;
        yEnd = im.height;
    }
    // now do a search
    for(int t = tStart; t<tEnd; t++) {
        for(int y = yStart; y<yEnd; y++) {
            for(int x = xStart; x<xEnd; x++) {
                fx = x; fy = y; ft = t;
                value  =  im(x, y, t)[0];
                // eliminate if not a x maximum
                if( xCheck && (im(x, y, t)[0] <= im(x-1, y, t)[0] ||
                               im(x, y, t)[0] <= im(x+1, y, t)[0]) ) {
                    continue;
                } else if (xCheck) {
                    // fine tune x coordinate by taking local centroid
                    fx += (im(x+1, y, t)[0]-im(x-1, y, t)[0])/(im(x, y, t)[0]+im(x-1, y, t)[0]+im(x+1, y, t)[0]);
                }
                // eliminate if not a y maximum
                if( yCheck && (im(x, y, t)[0] <= im(x, y-1, t)[0] ||
                               im(x, y, t)[0] <= im(x, y+1, t)[0]) ) {
                    continue;
                } else if (yCheck) {
                    // fine tune y coordinate by taking local centroid
                    fy += (im(x, y+1, t)[0]-im(x, y-1, t)[0])/(im(x, y, t)[0]+im(x, y-1, t)[0]+im(x, y+1, t)[0]);
                }
                // eliminate if not a t maximum
                if( tCheck && (im(x, y, t)[0] <= im(x, y, t-1)[0] ||
                               im(x, y, t)[0] <= im(x, y, t+1)[0]) ) {
                    continue;
                } else if (tCheck) {
                    // fine tune t coordinate by taking local centroid
                    ft += (im(x, y, t+1)[0]-im(x, y, t-1)[0])/(im(x, y, t)[0]+im(x, y, t-1)[0]+im(x, y, t+1)[0]);
                }
                // eliminate if not high enough
                float strength = threshold+1;
                if (xCheck) {
                    strength = min(strength, value - im(x-1, y, t)[0]);
                    strength = min(strength, value - im(x+1, y, t)[0]);
                } 
                if (yCheck) {
                    strength = min(strength, value - im(x, y+1, t)[0]);
                    strength = min(strength, value - im(x, y-1, t)[0]);
                } 
                if (tCheck) {
                    strength = min(strength, value - im(x, y, t+1)[0]);
                    strength = min(strength, value - im(x, y, t-1)[0]);
                } 
                if (strength < threshold) continue;

                // output if it is a candidate
                Maximum m;
                m.t = ft;
                m.x = fx;
                m.y = fy;
                m.value = value; 
                results.push_back(m);
            }
        }
    }

    if (minDistance < 1) return results;


    vector<LocalMaximaCollision> collisions;

    // Now search for collisions. This is made somewhat easier because
    // the results are already sorted by their t, then y, then x
    // coordinate.
    for (unsigned i = 0; i < results.size(); i++) {
        for (unsigned j = i+1; j < results.size(); j++) {
            float dist = 0, d;
            if (xCheck) {
                d = results[i].x - results[j].x;
                dist += d*d;
            }
            if (yCheck) {
                d = results[i].y - results[j].y;
                dist += d*d;
            } 
            if (tCheck) {
                d = results[i].t - results[j].t;
                dist += d*d;
            }
            
            if (dist < minDistance*minDistance) {
                LocalMaximaCollision c;
                if (results[i].value > results[j].value) {
                    c.disparity = results[i].value - results[j].value;
                    c.a = i;
                    c.b = j;
                } else {
                    c.disparity = results[j].value - results[i].value;
                    c.a = j;
                    c.b = i;                    
                }
                collisions.push_back(c);
            }

            // Early bailout. The +2 is because results may have
            // shifted by up to 1 each from their original sorted
            // locations due to the centroid finding above.
            if ((results[j].t - results[i].t) > (minDistance+2)) break;
            if (!tCheck && (results[j].y - results[i].y) > (minDistance+2)) break;
            if (!yCheck && !tCheck && (results[j].x - results[i].x) > (minDistance+2)) break;
        }
    }

    // Order the collisions from maximum strength disparity to minimum (i.e from easy decisions to hard ones)    
        ::std::sort(collisions.begin(), collisions.end());

    // Start by accepting them all, and knock some out greedily using
    // the collisions. This is a heurstic. To do this perfectly is a
    // multi-dimensional knapsack problem, and is NP-complete.
    vector<bool> accepted(results.size(), true);

    for (unsigned i = 0; i < collisions.size(); i++) {
        if (accepted[collisions[i].a] && accepted[collisions[i].b]) {
            accepted[collisions[i].b] = false;
        }
    }

    // return only the accepted points
    vector<LocalMaxima::Maximum> goodResults;
    for (unsigned i = 0; i < results.size(); i++) {
        if (accepted[i])
            goodResults.push_back(results[i]);
    }

    /*
    // draw the results on an image for debugging 
    for (unsigned i = 0; i < results.size(); i++) {
        int t = (int)(results[i].t+0.5);
        int y = (int)(results[i].y+0.5);
        int x = (int)(results[i].x+0.5);
        if (x < 0) x = 0; 
        if (x >= im.width) x = im.width-1;
        if (y < 0) y = 0;
        if (y >= im.height) y = im.height-1;
        if (t < 0) t = 0;
        if (t >= im.frames) t = im.frames-1;
        if (accepted[i]) im(x, y, t)[0] = 100;
        else im(x, y, t)[0] = -100;        
    }
    */

    return goodResults;
}


void Printf::help() {
    pprintf("-printf evaluates and prints its arguments, using the first argument"
            " as a format string. The remaining arguments are all evaluated as"
            " floats, so use %%d, %%i, and other non-float formats with caution.\n"
            "\n"
            "Usage: ImageStack -load foo.jpg -printf \"Mean  =  %%f\" \"mean()\"\n\n");
}

void Printf::parse(vector<string> args) {
    assert(args.size() > 0, "-printf requires at least one argument\n");
    vector<float> fargs;
    for (unsigned i = 1; i < args.size(); i++) {
        fargs.push_back(readFloat(args[args.size()-i]));
    }
    apply(stack(0), args[0], fargs);
}



void Printf::apply(Window im, string fmt, vector<float> a) {
    float args[16];

    assert(a.size() < 16, "-printf can't handle that many arguments\n");

    for (unsigned i = 0; i < a.size(); i++) {
        args[i] = a[i];
    }

    printf(fmt.c_str(), 
           args[0], args[1], args[2], args[3],
           args[4], args[5], args[6], args[7],
           args[8], args[9], args[10], args[11],
           args[12], args[13], args[14], args[15]);
    printf("\n");
}

void FPrintf::help() {
    pprintf("-fprintf evaluates and prints its arguments, appending them to the"
            " file specified by the first argument, using the second argument as a"
            " format string. The remaining arguments are all evaluated as floats,"
            " so use %%d, %%i, and other non-float formats with caution.\n\n"
            "Usage: ImageStack -load foo.jpg -fprintf results.txt \"Mean  =  %%f\" \"mean()\"");
}

void FPrintf::parse(vector<string> args) {
    assert(args.size() > 1, "-fprintf requires at least two arguments\n");
    vector<float> fargs;
    for (unsigned i = 2; i < args.size(); i++) {
        fargs.push_back(readFloat(args[args.size()-i+1]));
    }
    apply(stack(0), args[0], args[1], fargs);
}

void FPrintf::apply(Window im, string filename, string fmt, vector<float> a) {
    FILE *f = fopen(filename.c_str(), "a");
    assert(f, "Could not open %s\n", filename.c_str());

    float args[16];

    assert(a.size() < 16, "-printf can't handle that many arguments\n");

    for (unsigned i = 0; i < a.size(); i++) {
        args[i] = a[i];
    }

    fprintf(f, fmt.c_str(), 
            args[0], args[1], args[2], args[3],
            args[4], args[5], args[6], args[7],
            args[8], args[9], args[10], args[11],
            args[12], args[13], args[14], args[15]);
    fprintf(f, "\n");
    fclose(f);
}






void PCA::help() {
    pprintf("-pca reduces the number of channels in the image to the given"
            " parameter, using principal components analysis (PCA).\n\n"
            "Usage: ImageStack -load a.jpg -pca 1 -save gray.png\n");
}

void PCA::parse(vector<string> args) {
    assert(args.size() == 1, "-pca takes one argument\n");
    Image im = apply(stack(0), readInt(args[0]));
    pop();
    push(im);
}

Image PCA::apply(Window im, int newChannels) {
    assert(newChannels <= im.channels, "-pca can only reduce dimensionality, not expand it\n");

    Image out(im.width, im.height, im.frames, newChannels);

    Eigenvectors e(im.channels, out.channels);

    float *imPtr = im(0, 0);
    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                e.add(imPtr);
                imPtr += im.channels;
            }
        }
    }

    float *outPtr = out(0, 0);
    imPtr = im(0, 0);
    for (int t = 0; t < im.frames; t++) {
        for (int y = 0; y < im.height; y++) {
            for (int x = 0; x < im.width; x++) {
                e.apply(imPtr, outPtr);
                imPtr += im.channels;
                outPtr += out.channels;
            }
        }
    }

    return out;
    
}


void PatchPCA::help() {
    pprintf("-patchpca treats local Gaussian neighbourhoods of pixel values as vectors"
            " and computes a stack of filters that can be used to reduce"
            " dimensionality and decorrelate the color channels. The two arguments"
            " are the standard deviation of the Gaussian, and the desired number of"
            " output dimensions. Patches near the edge of the image are not"
            " included in the covariance computation.\n"
            "Usage: ImageStack -load a.jpg -patchpca 2 8 -save filters.tmp\n"
            " -pull 1 -convolve zero inner -save reduced.tmp\n");
}


void PatchPCA::parse(vector<string> args) {
    assert(args.size() == 2, "-patchpca takes two arguments\n");
    Image im = apply(stack(0), readFloat(args[0]), readInt(args[1]));
    push(im);
}


Image PatchPCA::apply(Window im, float sigma, int newChannels) {
    
    int patchSize = ((int)(sigma*6+1)) | 1;

    printf("Using %dx%d patches\n", patchSize, patchSize);

    vector<float> mask(patchSize);
    float sum = 0;
    printf("Gaussian mask: ");
    for (int i = 0; i < patchSize; i++) {
        mask[i] = expf(-(i-patchSize/2)*(i - patchSize/2)/(2*sigma*sigma));
        sum += mask[i];
        printf("%f ", mask[i]);
    }
    for (int i = 0; i < patchSize; i++) mask[i] /= sum;
    printf("\n");

    vector<float> vec(patchSize*patchSize*im.channels);

    Eigenvectors e(patchSize*patchSize*im.channels, newChannels);
    for (int iter = 0; iter < min(20000, im.width*im.height*im.frames); iter++) {
        int t = randomInt(0, im.frames-1);
        int x = randomInt(patchSize/2, im.width-1-patchSize/2);
        int y = randomInt(patchSize/2, im.height-1-patchSize/2);
        float *imPtr = im(x, y, t);
        int j = 0;
        for (int dy = -patchSize/2; dy <= patchSize/2; dy++) {
            for (int dx = -patchSize/2; dx <= patchSize/2; dx++) {
                for (int c = 0; c < im.channels; c++) {
                    vec[j] = (mask[dx+patchSize/2]*
                              mask[dy+patchSize/2]*
                              imPtr[dy*im.ystride + dx*im.xstride + c]);
                    j++;
                }
            }
        }
        e.add(&vec[0]);
    }

    e.compute();

    Image filters(patchSize, patchSize, 1, im.channels * newChannels);
    
    for (int i = 0; i < newChannels; i++) {
        e.getEigenvector(i, &vec[0]);
        int j = 0;
        for (int y = 0; y < patchSize; y++) {
            for (int x = 0; x < patchSize; x++) {
                for (int c = 0; c < im.channels; c++) {
                    filters(x, y)[i*im.channels+c] = vec[j];
                    j++;
                }
            }
        }
    }

    return filters;
}


#include "footer.h"
