# MATLAB_ROOT_DIR=/usr/sww/pkg/matlab-r2012b/
MATLAB_ROOT_DIR=/opt/matlab-r2012b/

BIN_CCFLAGS = -O3 -Winvalid-pch -Wall -pipe -fno-strict-aliasing -fomit-frame-pointer -fexpensive-optimizations -falign-functions=4 -funroll-loops -fprefetch-loop-arrays -march=nocona -msse3 -mfpmath=sse -Wno-uninitialized -I /usr/local/include

LIB_CCFLAGS = $(BIN_CCFLAGS) -fPIC

##########################
# Dependencies           #
##########################

# Comment out any dependencies you do not wish to include, or do not have
# SDL_CCFLAGS = -I /usr/include/SDL
# SDL_LIBS = `sdl-config --libs`

# JPEG_CCFLAGS =
# JPEG_LIBS = -ljpeg

# TIFF_CCFLAGS =
# TIFF_LIBS = -ltiff

# PNG_CCFLAGS = 
# PNG_LIBS = -lpng

# FFTW_CCFLAGS = 
# FFTW_LIBS = -lfftw3f

#OPENEXR_CCFLAGS = -I /usr/local/include/OpenEXR
#OPENEXR_LIBS = -L/usr/local/lib -lImath -lHalf -lIex -lIlmImf

#################################
# Target names and installation #
#################################

BIN_TARGET = bin/ImageStack
default: $(BIN_TARGET)
LIB_TARGET = lib/libImageStack.so
LIB_TARGET_MATLAB = lib/libImageStack.a
LIB_FLAGS = -shared -fPIC

install: $(BIN_TARGET)
	cp $(BIN_TARGET) /usr/local/bin

libinstall: library
	cp $(LIB_TARGET) /usr/local/lib
	mkdir -p /usr/local/include/ImageStack
	cp include/*.h /usr/local/include/ImageStack/

include Makefile.common











