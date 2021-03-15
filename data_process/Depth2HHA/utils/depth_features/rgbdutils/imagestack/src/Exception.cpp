#include "main.h"
#include "Exception.h"

#include "header.h"

Exception::Exception(const char *fmt, ...) {
    va_list arglist;
    va_start(arglist, fmt);
    vsnprintf(message, EXCEPTION_LENGTH, fmt, arglist);
    va_end(arglist);
}


Exception::Exception(const char *fmt, va_list arglist) {
    vsnprintf(message, EXCEPTION_LENGTH, fmt, arglist);
}

void panic(const char *fmt, ...) throw(Exception) {
    va_list arglist;
    va_start(arglist, fmt);
    Exception e(fmt, arglist);
    va_end(arglist);
    throw e;
}

void assert(bool cond, const char *fmt, ...) throw(Exception) {
    if (!cond) {
      va_list arglist;
      va_start(arglist, fmt);
      Exception e(fmt, arglist);
      va_end(arglist);
      throw e;
    }
}

#include "footer.h"
