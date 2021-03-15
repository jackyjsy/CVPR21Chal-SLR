#include "main.h"
#include "Image.h"
#include "header.h"

// The rest of the Image class is inlined.
// Inlining the destructor makes the compiler unhappy, so it goes here instead

// If you think there is a bug here, there is almost certainly
// actually a bug somewhere else which is corrupting the memory table
// or the image header. It will often crash here as a result because
// this is where things get freed.

Image::~Image() {    
    //printf("In image destructor\n"); fflush(stdout);

    if (!refCount) {
        //printf("Deleting NULL image\n"); fflush(stdout);
        return; // the image was a dummy
    }
    //printf("Decremementing refcount\n"); fflush(stdout);
    refCount[0]--;
    if (*refCount <= 0) {
        //printf("Deleting image\n"); fflush(stdout);
        //debug();
        delete refCount;
        //printf("refCount deleted\n"); fflush(stdout);
        //debug();        
        delete[] memory;
        //printf("data deleted\n"); fflush(stdout);
        //debug();
    }

    //printf("Leaving image desctructor\n"); fflush(stdout);
}
    
#include "footer.h"
