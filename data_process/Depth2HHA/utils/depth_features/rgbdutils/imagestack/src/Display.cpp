#include "main.h"
#include "Display.h"
#include "DisplayWindow.h"
#include "header.h"

#ifndef NO_SDL
Display::~Display() {
    DisplayWindow::instance()->wait();
}

void Display::help() {
    printf("\n-display opens a window and displays the current image. Subsequent displays\n"
           "will use the same window. The presence of an optional second argument indicates\n"
           "that the window should be fullscreen.\n\n"
           "Usage: ImageStack -load a.tga -loop 100 --display --gaussianblur 2\n\n");
    
}

void Display::parse(vector<string> args) {
    assert(args.size() < 2, "-display takes zero or one arguments\n");
    apply(stack(0), args.size() == 1);
}    
    
void Display::apply(Window im, bool fullscreen) {
    DisplayWindow::instance()->setMode(im.width, im.height, fullscreen);
    DisplayWindow::instance()->setImage(im);
       
    #ifdef __APPLE_CC__
    // OS X can't deal with launching a UI outside the main thread, so
    // we show it, wait until the user closes it, then continue
    DisplayWindow::instance()->show();
    #else
    // In the linux/windows case we show it in the background and
    // continue processing
    DisplayWindow::instance()->showAsync();
    #endif
}

#else
Display::~Display() {
}

void Display::help() {
    printf("This version of ImageStack was compiled without SDL, so cannot display.\n");
}

void Display::parse(vector<string> args) {
    panic("This version of ImageStack was compiled without SDL, so cannot display.\n");
}

void Display::apply(Window im, bool fullscreen) {
    panic("This version of ImageStack was compiled without SDL, so cannot display.\n");
}

#endif
#include "footer.h"
