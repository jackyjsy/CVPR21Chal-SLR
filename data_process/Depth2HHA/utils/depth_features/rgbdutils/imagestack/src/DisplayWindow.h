#ifndef IMAGESTACK_DISPLAYWINDOW_H
#define IMAGESTACK_DISPLAYWINDOW_H
#ifndef NO_SDL

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "main.h"
#include "Image.h"

#include <SDL.h>
#include <SDL_thread.h>
#include <SDL_mutex.h>

#include "header.h"

// a singleton SDL window
class DisplayWindow {
  public:
    static DisplayWindow *instance();
    void setMode(int width, int height, bool fullscreen = false, bool cursorVisible = true,
                 float bgRed = 0, float bgGreen = 0, float bgBlue = 0);

    int  width() {return width_;}
    int  height() {return height_;}
    bool fullscreen() {return fullscreen_;}

    void setImage(Window im);

    void setOffset(int tOffset, int xOffset, int yOffset);
    int  xOffset() {return xOffset_;}
    int  yOffset() {return yOffset_;}
    int  tOffset() {return tOffset_;}

    void show();
    void showAsync();
    void wait();

  private:
    DisplayWindow();
    ~DisplayWindow();

    static DisplayWindow *instance_;

    bool update();
    void redraw();
    void renderSurface();
    void updateCaption();
    void handleModeChange();
    bool terminate, modeChange, needRedraw;

    int width_, height_;
    bool fullscreen_, cursorVisible_;
    unsigned char bgRed_, bgGreen_, bgBlue_;
    int tOffset_, xOffset_, yOffset_;
    SDL_Surface *surface; 
    Uint8 *displayImage; // a 3D buffer of the 8bit image
    SDL_Thread *thread;
    SDL_mutex *mutex;
    int mouseX_, mouseY_;
    Image image_;
    int stop_;
};

#include "footer.h"
#endif
#endif
