// This should be included in every ImageStack .h and .cpp file,
// ideally after all the other includes. It places the rest of the
// file in the ImageStack namespace without encouraging a superfluous
// indent, and it uses a preprocessor counter to survive multiple
// inclusion without entering the namespace multiple times

#ifdef IMAGESTACK_HEADER_8_H
#error "Probable recursive inclusion of header.h"
#endif

#ifdef IMAGESTACK_HEADER_7_H
#define IMAGESTACK_HEADER_8_H
#endif

#ifdef IMAGESTACK_HEADER_6_H
#define IMAGESTACK_HEADER_7_H
#endif

#ifdef IMAGESTACK_HEADER_5_H
#define IMAGESTACK_HEADER_6_H
#endif

#ifdef IMAGESTACK_HEADER_4_H
#define IMAGESTACK_HEADER_5_H
#endif

#ifdef IMAGESTACK_HEADER_3_H
#define IMAGESTACK_HEADER_4_H
#endif

#ifdef IMAGESTACK_HEADER_2_H
#define IMAGESTACK_HEADER_3_H
#endif

#ifdef IMAGESTACK_HEADER_1_H
#define IMAGESTACK_HEADER_2_H
#endif

#ifndef IMAGESTACK_HEADER_1_H
#define IMAGESTACK_HEADER_1_H

// Included for the first time
namespace ImageStack {

#endif 

