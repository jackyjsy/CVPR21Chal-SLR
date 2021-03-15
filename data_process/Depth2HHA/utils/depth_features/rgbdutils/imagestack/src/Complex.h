#ifndef IMAGESTACK_COMPLEX_H
#define IMAGESTACK_COMPLEX_H
#include "header.h"

class ComplexMultiply : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window a, Window b, bool conj);
};

class ComplexDivide : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window a, Window b, bool conj);
};

class ComplexReal : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im);
};

class RealComplex : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im);
};

class ComplexImag : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im);
};

class ComplexMagnitude : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im);
};

class ComplexPhase : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static Image apply(Window im);
};

class ComplexConjugate : public Operation {
  public:
    void help();
    void parse(vector<string> args);
    static void apply(Window im);
};

#include "footer.h"
#endif
