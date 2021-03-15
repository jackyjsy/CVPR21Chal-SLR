#ifndef NO_SDL
#include "DisplayWindow.h"
#ifdef __CYGWIN__
#define WIN32
#endif
#include "main.h"
#include <SDL.h>
#include <SDL_thread.h>
#include <SDL_mutex.h>
#include <sstream>
#include "header.h"

DisplayWindow *DisplayWindow::instance() {    
    if (!instance_) {
        instance_ = new DisplayWindow();
    }
    return instance_;
}


DisplayWindow::DisplayWindow() {
    assert(instance_ == NULL, "A display has already been instantiated\n");

    surface = NULL;
    displayImage = NULL;
    width_ = height_ = 0;
    xOffset_ = yOffset_ = tOffset_ = 0;
    fullscreen_ = false;
    cursorVisible_ = true;

    stop_ = 0;

    modeChange = false;
    terminate = false;
    needRedraw = false;
    
    thread = NULL;

    mutex = SDL_CreateMutex();
}

DisplayWindow::~DisplayWindow() {
    if (displayImage) delete [] displayImage;
 
    SDL_mutexP(mutex);    
    if (surface) SDL_FreeSurface(surface);
    SDL_mutexV(mutex);

    if (thread) {
        SDL_WaitThread(thread, NULL);
    }

    instance_ = NULL;
    SDL_Quit();
}

void DisplayWindow::setMode(int width, int height, bool fullscreen, bool cursorVisible,
                            float bgRed, float bgGreen, float bgBlue) {

    cursorVisible_ = cursorVisible;
    // set the clear color
    bgRed_   = (unsigned char)(bgRed   * 255 + 0.499);
    bgGreen_ = (unsigned char)(bgGreen * 255 + 0.499);
    bgBlue_  = (unsigned char)(bgBlue  * 255 + 0.499);

    if (width == width_ && height == height_ && fullscreen == fullscreen_) return;

    width_ = width;
    height_ = height;
    fullscreen_ = fullscreen;

    modeChange = true;

    // wait on the thread
    if (thread) 
        while (modeChange) SDL_Delay(1);
}

void DisplayWindow::handleModeChange() {
    // set cursor status
    if (!cursorVisible_) SDL_ShowCursor(SDL_DISABLE);

    SDL_mutexP(mutex);
    SDL_Surface *result;
    if (fullscreen_) result = SDL_SetVideoMode(width_, height_, 32, SDL_DOUBLEBUF | SDL_FULLSCREEN);
    else result = SDL_SetVideoMode(width_, height_, 32, SDL_DOUBLEBUF);
    SDL_mutexV(mutex);
    assert(result != NULL, "Setting video mode failed: %s\n", SDL_GetError());

    SDL_WM_SetCaption("ImageStack Display Window","");

    // enable key repeat
    SDL_EnableKeyRepeat(SDL_DEFAULT_REPEAT_DELAY, SDL_DEFAULT_REPEAT_INTERVAL);
}

void DisplayWindow::setImage(Window im) {
    unsigned int rmask, gmask, bmask, amask;

    image_ = Image(im);

    #if SDL_BYTEORDER == SDL_BIG_ENDIAN
    rmask = 0xff000000;
    gmask = 0x00ff0000;
    bmask = 0x0000ff00;
    amask = 0x000000ff;
    #else
    rmask = 0x000000ff;
    gmask = 0x0000ff00;
    bmask = 0x00ff0000;
    amask = 0xff000000;
    #endif

    SDL_mutexP(mutex);
    if (surface) {
        // free the previous surface
        SDL_FreeSurface(surface);
    }

    surface = SDL_CreateRGBSurface(SDL_SWSURFACE, im.width, im.height, 32, rmask, gmask, bmask, amask);
    if (!surface) {
        SDL_mutexV(mutex);
        panic("Unable to allocate SDL surface: %s\n", SDL_GetError());
    }

    renderSurface();

    SDL_mutexV(mutex);
    needRedraw = true;
}

void DisplayWindow::renderSurface() {

    int SDL_y = 0;
    
    float scale = powf(2, stop_);

    while (tOffset_ < 0) tOffset_ += image_.frames;
    while (tOffset_ >= image_.frames) tOffset_ -= image_.frames;

    SDL_LockSurface(surface);

    switch(image_.channels) {

    case 1:
        for (int y = 0; y < image_.height; y++, SDL_y++) {
            Uint8 *scanlinePtr = (Uint8 *)surface->pixels + SDL_y*surface->pitch;
            float *imagePtr = image_(0, y, tOffset_);    
            for (int x = 0; x < image_.width; x++) {
                float val = imagePtr[0]; 
                *scanlinePtr++ = HDRtoLDR(scale * val);
                *scanlinePtr++ = HDRtoLDR(scale * val);
                *scanlinePtr++ = HDRtoLDR(scale * val);                        
                *scanlinePtr++ = 255;
                imagePtr ++;
            }
        }
        break;
    case 2:
        for (int y = 0; y < image_.height; y++, SDL_y++) {
            Uint8 *scanlinePtr = (Uint8 *)surface->pixels + SDL_y*surface->pitch;
            float *imagePtr = image_(0, y, tOffset_);    
            for (int x = 0; x < image_.width; x++) {
                *scanlinePtr++ = HDRtoLDR(scale*(*imagePtr++));
                *scanlinePtr++ = 0;
                *scanlinePtr++ = HDRtoLDR(scale*(*imagePtr++));
                *scanlinePtr++ = 255;
            }
        }
        break;

    case 3:
        for (int y = 0; y < image_.height; y++, SDL_y++) {
            Uint8 *scanlinePtr = (Uint8 *)surface->pixels + SDL_y*surface->pitch;
            float *imagePtr = image_(0, y, tOffset_);    
            for (int x = 0; x < image_.width; x++) {
                *scanlinePtr++ = HDRtoLDR(scale*(*imagePtr++));
                *scanlinePtr++ = HDRtoLDR(scale*(*imagePtr++));
                *scanlinePtr++ = HDRtoLDR(scale*(*imagePtr++));
                *scanlinePtr++ = 255;
            }
        }
        break;
    default:
        for (int y = 0; y < image_.height; y++, SDL_y++) {
            Uint8 *scanlinePtr = (Uint8 *)surface->pixels + SDL_y*surface->pitch;
            float *imagePtr = image_(0, y, tOffset_);    
            for (int x = 0; x < image_.width; x++) {
                *scanlinePtr++ = HDRtoLDR(imagePtr[0]*scale);
                *scanlinePtr++ = HDRtoLDR(imagePtr[1]*scale);
                *scanlinePtr++ = HDRtoLDR(imagePtr[2]*scale);
                *scanlinePtr++ = 255;
                imagePtr += image_.channels;
            }
        }
        break;
    }

    SDL_UnlockSurface(surface);
}

void DisplayWindow::setOffset(int tOffset, int xOffset, int yOffset) {
    tOffset_ = tOffset;
    xOffset_ = xOffset;
    yOffset_ = yOffset;
}

void DisplayWindow::redraw() {

    // fix the offsets
    if (!image_.frames)
        return;

    if (!surface) return;

    // copy the data over to staging surface

    SDL_Rect srcpos;
    srcpos.x = 0;
    srcpos.y = 0;
    srcpos.w = width_;
    srcpos.h = height_;

    // get the video surface
    SDL_Surface *screenbuffer = SDL_GetVideoSurface();
    assert(screenbuffer, "Unable to get screen buffer: %s\n", SDL_GetError());

    // clear the screen buffer
    assert((SDL_FillRect(screenbuffer, NULL, SDL_MapRGBA(screenbuffer->format, bgRed_, bgGreen_, bgBlue_, 0xff)) >= 0),
           "Clearing the screen failed: %s\n", SDL_GetError());

    SDL_Rect dstpos;
    dstpos.x = xOffset_;
    dstpos.y = yOffset_;
    dstpos.w = width_;
    dstpos.h = height_;

    // blit from staging surface to screen buffer surface
    assert((SDL_BlitSurface(surface, &srcpos, screenbuffer, &dstpos) >= 0), 
           "Blit failed: %s\n", SDL_GetError());

    // flip
    SDL_Flip(screenbuffer);

    // update the caption
    updateCaption();
}

void DisplayWindow::updateCaption() {
    int x = mouseX_ - xOffset_;
    int y = mouseY_ - yOffset_;
    int t = tOffset_;
    std::stringstream cap;
    cap << "ImageStack (x2^" << stop_ << ") (" << t << ", " << x << ", " << y << ")";

    if (x >= 0 && x < image_.width && y >= 0 && y < image_.height) {
        cap << " = (" << image_(x, y, t)[0];        
        for (int c = 1; c < image_.channels; c++) {
            cap << ", " << image_(x, y, t)[c];
        }
        cap << ")";
    }

    SDL_WM_SetCaption(cap.str().c_str(), "");
}

void DisplayWindow::show() {
    assert(SDL_Init(SDL_INIT_VIDEO) >= 0, "SDL failed to initialize: %s\n", SDL_GetError());

    while (1) {
        if (!update()) return;
        SDL_Delay(10);
    }
}

int threadFunc(void *data) {
    DisplayWindow::instance()->show();
    return 0;
}

void DisplayWindow::showAsync() {
    if (thread) return;
    terminate = false;
    thread = SDL_CreateThread(threadFunc, NULL);    
}

bool DisplayWindow::update() {
    // check for termination
    if (terminate) return false;

    // check for any pending display mode changes
    if (modeChange) {
        handleModeChange();
        modeChange = false;
    }
    
    // handle SDL events
    bool closeDisplayWindow = false;
    SDL_Event event;
    
    SDL_mutexP(mutex);
    while (SDL_PollEvent(&event)) {
        switch(event.type) {
        case SDL_QUIT:
            closeDisplayWindow = true;
            break;
        case SDL_KEYDOWN:
            switch(event.key.keysym.sym) {
            case SDLK_r:
                needRedraw = true;
                break;
            case SDLK_q:
            case SDLK_ESCAPE:
                closeDisplayWindow = true;
                break;
            case SDLK_UP:
                yOffset_-=100;
                needRedraw = true;
                break;
            case SDLK_DOWN:
                yOffset_+=100;
                needRedraw = true;
                break;
            case SDLK_LEFT:
                xOffset_-=100;
                needRedraw = true;
                break;
            case SDLK_RIGHT:
                xOffset_+=100;
                needRedraw = true;
                break;
            case SDLK_PAGEDOWN:
                tOffset_++;
                renderSurface();
                needRedraw = true;
                break;
            case SDLK_PAGEUP:
                tOffset_--;
                renderSurface();
                needRedraw = true;
                break;
            case SDLK_KP_PLUS:
            case SDLK_EQUALS:
                stop_ ++;
                renderSurface();
                needRedraw = true;
                break;
            case SDLK_KP_MINUS:
            case SDLK_MINUS:
                stop_ --;
                renderSurface();
                needRedraw = true;
                break;
            default:
                break;
            }
            break;
        case SDL_MOUSEMOTION:
            mouseX_ = event.motion.x;
            mouseY_ = event.motion.y;
            updateCaption();
            break;
        }
    }
    
    if (needRedraw) {
        redraw();
        needRedraw = false;
    } 

    SDL_mutexV(mutex);
    
    if (closeDisplayWindow) {
        thread = NULL;
        delete this;
        return false;
    }        

    return true;
}

void DisplayWindow::wait() {
    if (thread) {
        SDL_WaitThread(thread, NULL);
    }
}

DisplayWindow *DisplayWindow::instance_ = NULL;


#include "footer.h"
#endif

