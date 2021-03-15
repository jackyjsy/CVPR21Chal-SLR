# this file makes all look up tables needed by ImageStack

from math import *

f = open("tables.h", 'w')

f.write("#ifndef TABLES_H\n")
f.write("#define TABLES_H\n")
f.write("#include <math.h>\n")
f.write("#include \"header.h\"\n")

# lanczos filters of various widths
accuracy = 1024
for lobes in [2, 3, 4]:
    name = "lanczos_%i" % lobes
    size = (lobes + 1) * accuracy * 2
    f.write("static float _%s[%i] = {" % (name, size))
    for i in xrange(size):
        x = (i - size/2 + 0.5) / float(accuracy)
        x *= pi
        y = sin(x)/x * sin(x/lobes)/(x/lobes)
        if sin(x/lobes)/(x/lobes) < 0: y = 0
        f.write(`y`)
        if y != 0: f.write('f')
        if i < size-1: f.write(', ')
    f.write("};\n\n")

    f.write("static inline float %s(float x) {\n" % name)
    f.write("    return _%s[(int)(x * %i.0) + %i];\n" % (name, accuracy, size/2))
    f.write("}\n\n")


# precomputed exp function for [-10, 10]
size = 4096
f.write("static float _fastexp[%i] = {" % size)
for i in xrange(size):
    val = exp(i/float(size)*20 - 10)
    f.write(`val`)
    if val != 0: f.write('f')
    if i < size -1: f.write(', ')
f.write("};\n\n")
f.write("static inline float fastexp(float x) {\n")
f.write("    if (x < -9) return exp(x);")
f.write("    if (x > 9) return exp(x);")
f.write("    return _fastexp[(int)(x * 0.05f * %i) + %i];" % (size, size/2))
f.write("}\n\n")

f.write("#include \"footer.h\"\n")
f.write("#endif\n")


