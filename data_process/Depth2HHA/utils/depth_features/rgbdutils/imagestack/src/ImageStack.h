#ifndef IMAGESTACK_IMAGESTACK_H
#define IMAGESTACK_IMAGESTACK_H

// We never want SDL when used as a library
#define NO_SDL

// includes that don't survive well in the namespace
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <list>
#include <set>

#ifdef WIN32
#include <winsock2.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <netdb.h>
#endif

#include <math.h>

#include "main.h"
#include "Operation.h"
#include "Calculus.h"
#include "Color.h"
#include "Control.h"
#include "Convolve.h"
#include "Complex.h"
#include "DFT.h"
#include "Display.h"
#include "DisplayWindow.h"
#include "Exception.h"
#include "File.h"
#include "Filter.h"
#include "Geometry.h"
#include "GaussTransform.h"
#include "HDR.h"
#include "Image.h"
#include "LightField.h"
#include "Arithmetic.h"
#include "Network.h"
#include "NetworkOps.h"
#include "Paint.h"
#include "Panorama.h"
#include "Parser.h"
#include "Prediction.h"
#include "Projection.h"
#include "Stack.h"
#include "Statistics.h"
#include "Wavelet.h"
#include "WLS.h"
#include "macros.h"
#include "tables.h"

#undef NO_SDL

#endif
