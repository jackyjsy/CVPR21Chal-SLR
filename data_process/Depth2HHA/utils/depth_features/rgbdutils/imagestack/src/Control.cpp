#include "main.h"
#include "Control.h"
#include "header.h"

void Loop::help() {
    printf("\n-loop takes an integer and a sequence of commands, and loops that sequence\n"
           "the specified number of times. The commands that form the argument must be\n"
           "prefixed with an extra dash. It is possible to nest this operation using more\n"
           "dashes. If given no integer argument, loop will loop forever.\n\n"
           "Usage: ImageStack -load a.tga -loop 36 --rotate 10 --loop 10 ---downsample\n"
           "                  ---upsample -save b.tga\n\n");
}

void Loop::parse(vector<string> args) {
    assert(args.size() > 0, "-loop requires arguments\n");

    if (args[0].size() > 2 && args[0][0] == '-' && args[0][1] == '-') { // infinite loop mode
        vector<string> newArgs(args.size());        

        for (size_t i = 0; i < args.size(); i++) {
            if (args[i].size() > 1 && args[i][0] == '-' && args[i][1] == '-') {
                newArgs[i] = args[i].substr(1, args[i].size() - 1);
            } else {
                newArgs[i] = args[i];
            }
        }
        
        for (;;) parseCommands(newArgs);

    } else { // finite loop mode

        vector<string> newArgs(args.size() - 1);        
        
        for (size_t i = 0; i < newArgs.size(); i++) {
            if (args[i+1].size() > 1 && args[i+1][0] == '-' && args[i+1][1] == '-') {
                newArgs[i] = args[i+1].substr(1, args[i+1].size() - 1);
            } else {
                newArgs[i] = args[i+1];
            }
        }
        
        for (int i = 0; i < readInt(args[0]); i++) {
            parseCommands(newArgs);
        }
    }
}



void Pause::help() {
    printf("\n-pause waits for the user to press hit enter.\n\n"
           "Usage: ImageStack -load a.tga -display -pause -load b.tga -display\n\n");
}

void Pause::parse(vector<string> args) {
    assert(args.size() == 0, "-pause takes no arguments\n");
    fprintf(stdout, "Press enter to continue\n");
    char c = ' ';
    while (c != '\n' && c != EOF) c = getchar();
}

void Time::help() {
    printf("\n-time takes a sequence of commands, performs that sequence, and reports how\n"
           "long it took. The commands that form the argument must be prefixed with an extra\n"
           "dash. If given to arguments, it simply reports the time since the program was\n"
           "launched. It is a useful operation for profiling.\n\n"
           "Usage: ImageStack -load a.jpg -time --resample 10 10 --scale 2\n\n");
}

void Time::parse(vector<string> args) {
    if (args.size() == 0) {
        printf("%3.3f s\n", currentTime());
        return;
    }

    vector<string> newArgs(args.size());        
    
    for (size_t i = 0; i < args.size(); i++) {
        if (args[i].size() > 1 && args[i][0] == '-' && args[i][1] == '-') {
            newArgs[i] = args[i].substr(1, args[i].size() - 1);
        } else {
            newArgs[i] = args[i];
        }
    }
        
    float t1 = currentTime();
    parseCommands(newArgs);
    float t2 = currentTime();
    printf("%3.3f s\n", t2 - t1);
}

#include "footer.h"



