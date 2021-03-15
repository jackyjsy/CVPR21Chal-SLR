#include <ImageStack.h>

using namespace ImageStack;

// An example program that uses the ImageStack library to load a file, do some image processing, display it, and save it

int main(int argc, const char **argv) {
	if (argc != 3) {
		printf("Usage: example input.jpg output.jpg\n");
		return -1;
	}

	// Load an image
	Image im = Load::apply(argv[1]);

	// Do some image processing to it manually (inverting it)
	for (int t = 0; t < im.frames; t++) {
		for (int y = 0; y < im.height; y++) {
			for (int x = 0; x < im.width; x++) {
				for (int c = 0; c < im.channels; c++) {
					im(x, y, t)[c] = 1 - im(x, y, t)[c];
				}
			}
		}
	}

	// Call one of ImageStack's operators to smooth it
	im = WLS::apply(im, 1.2, 0.25, 0.01);

	// Save it out
	Save::apply(im, argv[2]);
	
	// Display it
	Display::apply(im);

	// Wait for input
	getchar();

	return 0;
}