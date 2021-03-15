#include "main.h"
#include "Stack.h"
#include "header.h"

void Pop::help() {
    printf("\n-pop removes the top image from the stack\n\n"
           "Usage: ImageStack -load a.tga -load b.tga -save b.gif -pop -save a.gif\n\n");
}

void Pop::parse(vector<string> args) {
    assert(args.size() == 0, "-pop takes no arguments\n");
    pop();
}

void Push::help() {
    printf("-push adds a new zeroed image to the top of the stack. With no"
           " arguments it matches the dimensions of the current image. With 4"
           " arguments (width, height, frames, and channels) it creates an image of"
           " that size. Given three arguments frames defaults to 1, and the"
           " arguments are taken as width, height, and channels.\n"
           "\n"
           "Usage: ImageStack -load a.tga -push -add -scale 0.5 -multiply -save out.tga\n"
           "       ImageStack -push 1024 1024 1 3 -offset 0.5 -save gray.tga\n\n");
}

void Push::parse(vector<string> args) {
    if (args.size() == 0) {
        push(Image(stack(0).width, stack(0).height, stack(0).frames, stack(0).channels));
    } else if (args.size() == 3) {
        push(Image(readInt(args[0]), readInt(args[1]), 1, readInt(args[2])));
    } else if (args.size() == 4) {
        push(Image(readInt(args[0]), readInt(args[1]), readInt(args[2]), readInt(args[3])));
    } else {
        panic("-push takes zero, three, or four arguments\n");
    }
}

void Pull::help() {
    printf("\n-pull brings a buried stack element to the top. -pull 0 does nothing. -pull 1\n"
           "brings up the second stack element, and so on.\n\n"
           "Usage: ImageStack -load a.tga -load b.tga -save b.gif -pull 1 -save a.gif\n\n");
}

void Pull::parse(vector<string> args) {
    assert(args.size() == 1, "-pull takes 1 argument\n");
    int depth = readInt(args[0]);
    assert(depth > 0, "-pull only makes sense on strictly positive depths\n");
    pull(depth);
}

void Dup::help() {
    printf("\n-dup duplicates the current image and pushes it on the stack.\n\n"
           "Usage: ImageStack -load a.tga -dup -scale 0.5 -save a_small.tga\n"
           "                  -pop -scale 2 -save a_big.tga\n\n");
        
}

void Dup::parse(vector<string> args) {
    assert(args.size() == 0, "-dup takes no arguments\n");    
    dup();
}

#include "footer.h"
