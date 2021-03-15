
#include "main.h"
#include "time.h"
#include "Parser.h"
#ifndef WIN32
#include <sys/time.h>
#endif
#include "header.h"

vector<Image> stack_;
Image &stack(size_t idx) {
    assert(idx < stack_.size(), "Stack underflow\n");
    return stack_[stack_.size() - 1 - idx];
}

void push(Image im) {
    stack_.push_back(im);
}

void pop() {
    assert(stack_.size(), "Stack underflow\n");
    stack_.pop_back();
}

void dup() {
    Image &top = stack(0);
    Image newTop(top.width, top.height, top.frames, top.channels);
    memcpy(newTop.data, top.data, top.frames * top.width * top.height * top.channels * sizeof(float));
    stack_.push_back(newTop);
}

void pull(size_t n) {
    assert(n < stack_.size(), "Stack underflow\n");
    for (size_t i = stack_.size() - n - 1; i < stack_.size()-1; i++) {
        Image tmp = stack_[i+1];
        stack_[i+1] = stack_[i];
        stack_[i] = tmp;
    }
}

int randomInt(int min, int max) {
    return (int)(((double)rand()/(RAND_MAX+1.0)) * (max - min + 1) + min);
}

float randomFloat(float min, float max) {
    return ((float)rand()/(RAND_MAX+1.0)) * (max - min) + min;
}

#ifdef WIN32
DWORD startTime;
float currentTime() {
    DWORD now = timeGetTime();
    return (now - startTime) * 0.001f;
}
#else
struct timeval startTime;
float currentTime() {
    struct timeval now;
    gettimeofday(&now, NULL);
    return (now.tv_sec - startTime.tv_sec) + (now.tv_usec - startTime.tv_usec) / 1000000.0f;
}
#endif

map<string, Operation *> operationMap;

void start() {
    // get the starting time
#ifdef WIN32
    startTime = timeGetTime();
    srand(startTime); rand();
#else
    gettimeofday(&startTime, NULL);
    srand(startTime.tv_sec + startTime.tv_usec); rand();
#endif
    // make the operation map
    loadOperations();
}

void end() {
    unloadOperations();
}

void parseCommands(vector<string> args) {
    size_t arg = 0, opArgs;
    OperationMapIterator op;

    while (arg < args.size()) {
        // dump the stack for debugging
        /*
        if (0) {
            printf("Stack: \n");
            for (size_t i = 0; i < stack_.size(); i++) {
                stack_[i].debug();
            }
        }
        */

        // get the operation
        op = operationMap.find(args[arg]);

        // check the op is exists
        if (op == operationMap.end()) {
            panic("Unknown operation \"%s\"\n"
                  "Try -help for a list of operations.", args[arg].c_str());
        }

        // find the arguments, look ahead till we see -[a-zA-Z]
        for (opArgs = 1; opArgs + arg < args.size(); opArgs++) {
            char first = args[arg + opArgs][0];
            assert(first != '\0', "Empty argument!");
            if (first != '-') continue;
            if (isalpha(args[arg + opArgs][1])) break;
        }

        printf("Performing operation %s ", op->first.c_str()); fflush(stdout);
        if (opArgs < 8) {
            for (size_t i = arg+1; i < arg + opArgs; i++) {
                printf("%s ", args[i].c_str());
            }
        }
        printf("...\n");

        vector<string> operationArgs;
        for (size_t i = arg + 1; i < arg + opArgs; i++) operationArgs.push_back(args[i]);

        // call the operation
        (op->second)->parse(operationArgs);

        // skip over the args
        arg += opArgs;
    }
}


int readInt(string arg) {
    return (int)(floorf(readFloat(arg)+0.5f));
}


float readFloat(string arg) {
    bool needToPop = false;
    Expression e(arg, false);
    if (stack_.size() == 0) {
        push(Image(1, 1, 1, 1));
        needToPop = true;
    }
    Expression::State s(stack(0));
    float val = e.eval(&s);    
    if (needToPop) pop();
    return val;
}


char readChar(string arg) {
    assert(arg.size() == 1,
           "Argument '%s' is not a single character\n", arg.c_str());
    return arg[0];
}


// pretty-print some help text, by word wrapping at 80 chars
void pprintf(const char *str) {    
    const char *startOfLine = str;
    const char *endOfLine = str;
    char line[81];
    char *linePtr;

    while (*endOfLine) {
        // walk ahead to 80 characters or EOL, whichever comes first
        endOfLine = startOfLine;
        linePtr = line;
        while (endOfLine - startOfLine < 80 && *endOfLine && *endOfLine != '\n') {
            *linePtr++ = *endOfLine++;
        }
        
        if (!*endOfLine) {
            *linePtr = 0;
            printf("%s", line);
            return;
        }

        if (*endOfLine == '\n') {
            linePtr[0] = '\n';
            linePtr[1] = 0;
            printf("%s", line);
            startOfLine = endOfLine + 1;
            continue;
        }

        // walk back to the last space
        char *lastSpace = linePtr;
        while (*lastSpace != ' ' && lastSpace > line + 40) {
            lastSpace--;
        }

        // if we found one, print up to it
        if (*lastSpace == ' ') {
            lastSpace[0] = '\n';
            lastSpace[1] = 0;
            printf("%s", line);
            startOfLine += lastSpace - line + 1;
        } 
    }
}
#include "footer.h"

#ifndef NO_MAIN

// We need to make sure main gets replaced with SDL_main on OS X, or
// display won't work properly. On other platforms it's not necessary,
// and might slow down non-display using ImageStack command lines, so
// we don't do it.
#ifdef __APPLE_CC__ 
#ifndef NO_SDL
#include <SDL.h>
#endif
#endif

using namespace ImageStack;

int main(int argc, char **argv) {

    start();

    if (argc == 1 || argv[1][0] != '-') {
        operationMap["-help"]->help();
    }

    vector<string> args;
    for (int i = 1; i < argc; i++) {
        args.push_back(argv[i]);
    }

    try {
        parseCommands(args);
    } catch(Exception &e) {
        printf("%s", e.message);
    }

    fflush(stdout);
    fflush(stderr);

    end();

    return 0;

}

#endif
