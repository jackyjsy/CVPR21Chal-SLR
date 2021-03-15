#include "main.h"
#include "Operation.h"

// Include your operation's source file or header here.
// It should must define a subclass of Operation (in Operation.h).

// You might want to look in Arithmetic.cpp/Arithmetic.h for some examples.

// Also check main.h for functions available to parse input, handle errors, etc.

#include "Control.h"
#include "Statistics.h"
#include "Arithmetic.h"
#include "Alignment.h"
#include "Stack.h"
#include "Geometry.h"
#include "Convolve.h"
#include "Color.h"
#include "Complex.h"
#include "LightField.h"
#include "Paint.h"
#include "File.h"
#include "Calculus.h"
#include "Display.h"
#include "NetworkOps.h"
#include "Panorama.h"
#include "DFT.h"
#include "Prediction.h"
#include "Wavelet.h"
#include "Filter.h"
#include "Projection.h"
#include "HDR.h"
#include "PatchMatch.h"
#include "GaussTransform.h"
#include "LAHBPCG.h"
#include "WLS.h"

#include "header.h"


// add your operation to the operations table here
void loadOperations() {
    operationMap["-help"] = new Help();

    // program control
    operationMap["-loop"] = new Loop();
    operationMap["-pause"] = new Pause();
    operationMap["-time"] = new Time();

    // statistics

    operationMap["-statistics"] = new Statistics();
    operationMap["-dimensionreduction"] = new DimensionReduction();
    operationMap["-dimensions"] = new Dimensions();
    operationMap["-noise"] = new Noise();
    operationMap["-histogram"] = new Histogram();
    operationMap["-equalize"] = new Equalize();
    operationMap["-histogrammatch"] = new HistogramMatch();
    operationMap["-shuffle"] = new Shuffle();
    operationMap["-kmeans"] = new KMeans();
    operationMap["-sort"] = new Sort();
    operationMap["-localmaxima"] = new LocalMaxima();
    operationMap["-printf"] = new Printf();
    operationMap["-fprintf"] = new FPrintf();
    operationMap["-pca"] = new PCA();
    operationMap["-patchpca"] = new PatchPCA();

    // file IO
    operationMap["-load"] = new Load();
    operationMap["-save"] = new Save();
    operationMap["-loadframes"] = new LoadFrames();
    operationMap["-saveframes"] = new SaveFrames();
    operationMap["-loadarray"] = new LoadArray();
    operationMap["-savearray"] = new SaveArray();
    operationMap["-loadblock"] = new LoadBlock();
    operationMap["-saveblock"] = new SaveBlock();
    operationMap["-createtmp"] = new CreateTmp();

    // basic maths
    operationMap["-add"] = new Add();
    operationMap["-multiply"] = new Multiply();
    operationMap["-subtract"] = new Subtract();
    operationMap["-divide"] = new Divide();
    operationMap["-max"] = new Maximum();
    operationMap["-min"] = new Minimum();
    operationMap["-log"] = new Log();
    operationMap["-exp"] = new Exp();
    operationMap["-offset"] = new Offset();
    operationMap["-scale"] = new Scale();
    operationMap["-gamma"] = new Gamma();
    operationMap["-mod"] = new Mod();
    operationMap["-normalize"] = new Normalize();
    operationMap["-clamp"] = new Clamp();
    operationMap["-denan"] = new DeNaN();
    operationMap["-threshold"] = new Threshold();
    operationMap["-abs"] = new Abs();
    operationMap["-quantize"] = new Quantize();

    // stack ops
    operationMap["-pop"] = new Pop();
    operationMap["-push"] = new Push();
    operationMap["-pull"] = new Pull();
    operationMap["-dup"] = new Dup();

    // geometry
    operationMap["-resample"] = new Resample();
    operationMap["-crop"] = new Crop();
    operationMap["-flip"] = new Flip();
    operationMap["-adjoin"] = new Adjoin();
    operationMap["-transpose"] = new Transpose();
    operationMap["-translate"] = new Translate();
    operationMap["-paste"] = new Paste();
    operationMap["-downsample"] = new Downsample();
    operationMap["-upsample"] = new Upsample();
    operationMap["-rotate"] = new Rotate();
    operationMap["-affinewarp"] = new AffineWarp();
    operationMap["-tile"] = new Tile();
    operationMap["-subsample"] = new Subsample();
    operationMap["-warp"] = new Warp();
    operationMap["-interleave"] = new Interleave();
    operationMap["-deinterleave"] = new Deinterleave();
    operationMap["-tileframes"] = new TileFrames();
    operationMap["-frametiles"] = new FrameTiles();
    operationMap["-reshape"] = new Reshape();

    // alignment
    operationMap["-align"] = new Align();
    operationMap["-alignframes"] = new AlignFrames();

    // color stuff
    operationMap["-colormatrix"] = new ColorMatrix();
    operationMap["-colorconvert"] = new ColorConvert();
    operationMap["-demosaic"] = new Demosaic();

    // convolutions
    operationMap["-convolve"] = new Convolve();
    operationMap["-deconvolve"] = new Deconvolve();

    // calculus
    operationMap["-gradient"] = new Gradient();
    operationMap["-integrate"] = new Integrate();
    operationMap["-gradmag"] = new GradMag();
    operationMap["-poisson"] = new Poisson();

    // complex number ops
    operationMap["-realcomplex"] = new RealComplex();
    operationMap["-complexmultiply"] = new ComplexMultiply();
    operationMap["-complexdivide"] = new ComplexDivide();
    operationMap["-complexreal"] = new ComplexReal();
    operationMap["-compleximag"] = new ComplexImag();
    operationMap["-complexconjugate"] = new ComplexConjugate();
    operationMap["-complexmagnitude"] = new ComplexMagnitude();
    operationMap["-complexphase"] = new ComplexPhase();
    

    #ifndef NO_FFTW
    // discrete fourier transforms
    operationMap["-dct"] = new DCT();
    operationMap["-fft"] = new FFT();
    operationMap["-ifft"] = new IFFT();
    operationMap["-fftconvolve"] = new FFTConvolve();
    operationMap["-fftdeconvolve"] = new FFTDeconvolve();
    operationMap["-fftpoisson"] = new FFTPoisson();
    #endif

    // painting stuff
    operationMap["-eval"] = new Eval();
    operationMap["-evalchannels"] = new EvalChannels();
    operationMap["-plot"] = new Plot();
    operationMap["-composite"] = new Composite();

    // lightfield stuff
    operationMap["-lffocalstack"] = new LFFocalStack();
    operationMap["-lfwarp"] = new LFWarp();
    operationMap["-lfpoint"] = new LFPoint();

    // the display code
    operationMap["-display"] = new Display();

    // network stuff
    operationMap["-send"] = new Send();
    operationMap["-receive"] = new Receive();

    // Panorama stuff
    operationMap["-loadpanorama"] = new LoadPanorama();
    operationMap["-panoramabackground"] = new PanoramaBackground();

    // prediction stuff
    operationMap["-inpaint"] = new Inpaint();

    // wavelet stuff
    operationMap["-haar"] = new Haar();
    operationMap["-inversehaar"] = new InverseHaar();
    operationMap["-daubechies"] = new Daubechies();
    operationMap["-inversedaubechies"] = new InverseDaubechies();

    // some filters
    operationMap["-gaussianblur"] = new GaussianBlur();
    operationMap["-lanczosblur"] = new LanczosBlur();
    operationMap["-fastblur"] = new FastBlur();
    operationMap["-rectfilter"] = new RectFilter();
    operationMap["-circularfilter"] = new CircularFilter();
    operationMap["-medianfilter"] = new MedianFilter();
    operationMap["-percentilefilter"] = new PercentileFilter();
    operationMap["-envelope"] = new Envelope();

    // Filters that use a Gauss transform
    operationMap["-gausstransform"] = new GaussTransform();
    operationMap["-bilateral"] = new Bilateral();
    operationMap["-jointbilateral"] = new JointBilateral();
    operationMap["-bilateralsharpen"] = new BilateralSharpen();
    operationMap["-chromablur"] = new ChromaBlur();
    operationMap["-nlmeans"] = new NLMeans();
   
    // Locally Adaptive Hierachical Basis Preconditioned Conjugate Gradients
    operationMap["-lahbpcg"] = new LAHBPCG();
    
    // Weighted-Least-Squares filtering
    operationMap["-wls"] = new WLS();

    // projection stuff
    operationMap["-sinugram"] = new Sinugram();
    
    // HDR stuff
    operationMap["-assemblehdr"] = new AssembleHDR();

    // PatchMatch stuff
    operationMap["-patchmatch"] = new PatchMatch();
    operationMap["-bidirectionalsimilarity"] = new BidirectionalSimilarity();
    operationMap["-heal"] = new Heal();
}


void unloadOperations() {
    OperationMapIterator i;
    for (i = operationMap.begin(); i != operationMap.end(); ++i) {
        delete i->second;
    }
}


void Help::help() {
    pprintf("ImageStack is a stack language for manipulating images. It"
            " is appropriate for use on the command line and in scripts."
            " Internally, all data is stored as 32 bit floating point, so"
            " ImageStack is good for high dynamic range data. ImageStack"
            " is also useful for low dynamic range data, which it treats as"
            " values between 0 and 1.\n\n"
            "-help provides help on a given operation.\n"
            "Usage: ImageStack -help scale\n\n"
            "Operations available are:\n");

    OperationMapIterator i;

    for (i = operationMap.begin(); i != operationMap.end(); ++i) {
        printf("%s", i->first.c_str());
        printf(" ");
    }
    printf("\n");
}


void Help::parse(vector<string> args) {
    if (args.size() < 1) {
        help();
    } else {
        string opname = '-' + args[0];
        OperationMapIterator op = operationMap.find(opname);
        if (op == operationMap.end())
            printf("No such operation \"%s\"\n", args[0].c_str());
        else {
            printf("\n");
            op->second->help();
        }
    }
}

#include "footer.h"
