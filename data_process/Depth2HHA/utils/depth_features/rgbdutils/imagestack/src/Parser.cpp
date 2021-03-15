#include "main.h"
#include "Parser.h"
#include "header.h"

void Expression::skipWhitespace() {
    while (source[sourceIndex] == ' ' || source[sourceIndex] == '\t' || source[sourceIndex] == '\n') sourceIndex++;
}

// Check ahead to see if the next few characters match 'prefix'
bool Expression::match(string prefix) {
    skipWhitespace();

    // check ahead
    size_t prefixPtr = 0;
    size_t sourcePtr = 0;
    for (;;) {
        if (prefixPtr == prefix.size()) {
            return true;
        } 
        if (prefix[prefixPtr] != source[sourceIndex + sourcePtr]) {
            return false;
        } 
        if (sourcePtr == source.size()) {
            return false;
        }
        prefixPtr++;
        sourcePtr++;
    }
}

// If the next few characters match 'prefix', increment the source
// pointer to beyond them are return true.
bool Expression::consume(string prefix) {
    if (match(prefix)) {
        sourceIndex += prefix.size();
        return true;
    } 
    return false;
}

// IfThenElse -> Condition (? Condition : Condition)?
Expression::Node *Expression::parseIfThenElse() {
    //printf("parsing if then else\n");
    Node *result = parseCondition();

    if (consume("?")) {
        Node *then = parseCondition();
        assert(consume(":"), "If Then Else missing else case\n");
        result = new IfThenElse(result, then, parseCondition());
    }

    return result;
}

// Condition -> Sum ((>|<|>=|<=|==|!=) Sum)?
Expression::Node *Expression::parseCondition() {
    //printf("parsing condition\n");
    Node *result = parseSum();

    if (consume("<=")) {
        result = new LTE(result, parseSum());
    } else if (consume(">=")) {
        result = new GTE(result, parseSum());
    } else if (consume("<")) {
        result = new LT(result, parseSum());
    } else if (consume(">")) {
        result = new GT(result, parseSum());
    } else if (consume("==")) {
        result = new EQ(result, parseSum());
    } else if (consume("!=")) {
        result = new NEQ(result, parseSum());
    } 
        
    return result;
}

// Sum     -> Product ((+|-) Product)*
Expression::Node *Expression::parseSum() {
    //printf("parsing sum\n");
    Node *result = parseProduct();

    for (;;) {
        if (consume("+")) {
            result = new Plus(result, parseProduct());
        } else if (consume("-")) {
            result = new Minus(result, parseProduct());
        } else {
            return result;
        }
    }
    //printf("done parsing sum\n");
}

// Product -> Factor ((*|/|%) Factor)*
Expression::Node *Expression::parseProduct() {
    //printf("parsing product\n");
    Node *result = parseFactor();

    for (;;) {
        if (consume("*")) {
            result = new Times(result, parseFactor());
        } else if (consume("/")) {
            result = new Divide(result, parseFactor());
        } else if (consume("%")) {
            result = new Mod(result, parseFactor());
        } else {
            //printf("done parsing product\n");
            return result;
        }
    }
}

// Factor  -> Term ^ Term | Term  
Expression::Node *Expression::parseFactor() {
    //printf("parsing factor\n");
    Node *result = parseTerm();
    if (consume("^")) {
        result = new Power(result, parseTerm());
    }
    //printf("done parsing factor\n");
    return result;
}

// Term    -> Funct ( IfThenElse ) | - Term | Var | ( IfThenElse ) | Float | Sample
Expression::Node *Expression::parseTerm() {
    //printf("parsing term\n");
    Node *result;

    //printf("%s\n", source);

    if (varyingAllowed && consume("[")) { // sampling
        Node *coord1 = parseIfThenElse();
        if (consume(",")) {
            Node *coord2 = parseIfThenElse();
            if (consume(",")) {
                Node *coord3 = parseIfThenElse();
                assert(consume("]"), "Sample takes at most three coordinates\n");
                result = new Sample3D(coord1, coord2, coord3);
            } else {
                result = new Sample2D(coord1, coord2);
                assert(consume("]"), "Sample missing closing bracket\n");
            }
        } else {
            result = new SampleHere(coord1);
            assert(consume("]"), "Sample missing closing bracket\n");
        }
    } else if (consume("cos")) { // functions
        assert(consume("("), "Function call missing opening parenthesis.\n");
        result = new Funct_cos(parseIfThenElse());
        assert(consume(")"), "Function call missing closing parenthesis.\n");
    } else if (consume("sin")) {
        assert(consume("("), "Function call missing opening parenthesis.\n");
        result = new Funct_sin(parseIfThenElse());
        assert(consume(")"), "Function call missing closing parenthesis.\n");
    } else if (consume("log")) {
        assert(consume("("), "Function call missing opening parenthesis.\n");
        result = new Funct_log(parseIfThenElse());
        assert(consume(")"), "Function call missing closing parenthesis.\n");        
    } else if (consume("exp")) {
        assert(consume("("), "Function call missing opening parenthesis.\n");
        result = new Funct_exp(parseIfThenElse());
        assert(consume(")"), "Function call missing closing parenthesis.\n");        
    } else if (consume("tan")) {
        assert(consume("("), "Function call missing opening parenthesis.\n");
        result = new Funct_tan(parseIfThenElse());
        assert(consume(")"), "Function call missing closing parenthesis.\n");            
    } else if (consume("atan")) {
        assert(consume("("), "Function call missing opening parenthesis.\n");
        result = new Funct_atan(parseIfThenElse());
        assert(consume(")"), "Function call missing closing parenthesis.\n");            
    } else if (consume("asin")) {
        assert(consume("("), "Function call missing opening parenthesis.\n");
        result = new Funct_asin(parseIfThenElse());
        assert(consume(")"), "Function call missing closing parenthesis.\n");            
    } else if (consume("acos")) {
        assert(consume("("), "Function call missing opening parenthesis.\n");
        result = new Funct_acos(parseIfThenElse());
        assert(consume(")"), "Function call missing closing parenthesis.\n");            
    } else if (consume("atan2")) {
        assert(consume("("), "Function call missing opening parenthesis.\n");
        Node *arg1 = parseIfThenElse();
        assert(consume(","), "',' expected between function call arguments.\n");
        Node *arg2 = parseIfThenElse();
        assert(consume(")"), "Function call missing closing parenthesis.\n");
        result = new Funct_atan2(arg1, arg2);
    } else if (consume("abs")) {
        assert(consume("("), "Function call missing opening parenthesis.\n");
        result = new Funct_abs(parseIfThenElse());
        assert(consume(")"), "Function call missing closing parenthesis.\n");            
    } else if (consume("floor")) {
        assert(consume("("), "Function call missing opening parenthesis.\n");
        result = new Funct_floor(parseIfThenElse());
        assert(consume(")"), "Function call missing closing parenthesis.\n");            
    } else if (consume("ceil")) {
        assert(consume("("), "Function call missing opening parenthesis.\n");
        result = new Funct_ceil(parseIfThenElse());
        assert(consume(")"), "Function call missing closing parenthesis.\n");            
    } else if (consume("round")) {
        assert(consume("("), "Function call missing opening parenthesis.\n");
        result = new Funct_round(parseIfThenElse());
        assert(consume(")"), "Function call missing closing parenthesis.\n");
    } else if (consume("mean")) {
        assert(consume("("), "Function call missing opening parenthesis.\n");
        if (consume(")")) {
            result = new Funct_mean0();
        } else {
            result = new Funct_mean1(parseIfThenElse());
            assert(consume(")"), "Function call missing closing parenthesis.\n");
        }
    } else if (consume("sum")) {
        assert(consume("("), "Function call missing opening parenthesis.\n");
        if (consume(")")) {
            result = new Funct_sum0();
        } else {
            result = new Funct_sum1(parseIfThenElse());
            assert(consume(")"), "Function call missing closing parenthesis.\n");
        }
    } else if (consume("max")) {
        assert(consume("("), "Function call missing opening parenthesis.\n");
        if (consume(")")) {
            result = new Funct_max0();
        } else {
            result = new Funct_max1(parseIfThenElse());
            assert(consume(")"), "Function call missing closing parenthesis.\n");
        }
    } else if (consume("min")) {
        assert(consume("("), "Function call missing opening parenthesis.\n");
        if (consume(")")) {
            result = new Funct_min0();
        } else {
            result = new Funct_min1(parseIfThenElse());
            assert(consume(")"), "Function call missing closing parenthesis.\n");
        }
    } else if (consume("stddev")) {
        assert(consume("("), "Function call missing opening parenthesis.\n");
        if (consume(")")) {
            result = new Funct_stddev0();
        } else {
            result = new Funct_stddev1(parseIfThenElse());
            assert(consume(")"), "Function call missing closing parenthesis.\n");
        }
    } else if (consume("variance")) {
        assert(consume("("), "Function call missing opening parenthesis.\n");
        if (consume(")")) {
            result = new Funct_variance0();
        } else {
            result = new Funct_variance1(parseIfThenElse());
            assert(consume(")"), "Function call missing closing parenthesis.\n");
        }
    } else if (consume("skew")) {
        assert(consume("("), "Function call missing opening parenthesis.\n");
        if (consume(")")) {
            result = new Funct_skew0();
        } else {
            result = new Funct_skew1(parseIfThenElse());
            assert(consume(")"), "Function call missing closing parenthesis.\n");
        }
    } else if (consume("kurtosis")) {
        assert(consume("("), "Function call missing opening parenthesis.\n");
        if (consume(")")) {
            result = new Funct_kurtosis0();
        } else {
            result = new Funct_kurtosis1(parseIfThenElse());
            assert(consume(")"), "Function call missing closing parenthesis.\n");
        }
    } else if (consume("covariance")) {
        assert(consume("("), "Function call missing opening parenthesis.\n");
        Node *arg1 = parseIfThenElse();
        assert(consume(","), "',' expected between function call arguments.\n");
        Node *arg2 = parseIfThenElse();        
        assert(consume(")"), "Function call missing closing parenthesis.\n");
        result = new Funct_covariance(arg1, arg2);
    } else if (consume("-")) { // unary negation
        result = new Negation(parseTerm());
    } else if (varyingAllowed && consume("x")) { // variables
        result = new Var_x();
    } else if (varyingAllowed && consume("y")) { 
        result = new Var_y();
    } else if (varyingAllowed && consume("t")) { 
        result = new Var_t();
    } else if (varyingAllowed && consume("c")) { 
        result = new Var_c();
    } else if (varyingAllowed && consume("val")) {
        result = new Var_val();
    } else if (consume("(")) { // a new expression bracketed
        result = parseIfThenElse();
        assert(consume(")"), "Closing parenthesis missing on expression\n");
    } else if (consume("pi")) { // constants
        result = new Float(3.14159265f);
    } else if (consume("e")) {
        result = new Float(2.71828183f);
    } else if (consume("width")) {
        result = new Uniform_width();
    } else if (consume("height")) {
        result = new Uniform_height();
    } else if (consume("frames")) {
        result = new Uniform_frames();
    } else if (consume("channels")) {
        result = new Uniform_channels();
    } else { // must be a float constant
        float val;
        int consumed = sscanf(source.c_str() + sourceIndex, "%f", &val);
        if (consumed == 0) panic("Could not parse from here: %s\n", source.c_str() + sourceIndex);
        result = new Float(val);
        // now we must skip the constant
        // fortunately, a float can't sensibly be followed by . or e or a digit
        while (isdigit(source[sourceIndex]) || source[sourceIndex] == '.' || source[sourceIndex] == 'e') sourceIndex++;
    }
    //printf("done parsing term\n");
    return result;
}


Expression::Expression(string source_, bool varyingAllowed_) {
    varyingAllowed = varyingAllowed_;
    source = source_;
    sourceIndex = 0;
    root = parseIfThenElse();
    skipWhitespace();
    assert(sourceIndex == source.size(), "Portion of expression not parsed: %s\n", source.c_str() + sourceIndex);
}

Expression::~Expression() {
    delete root;
}

float Expression::eval(State *state) {
    return root->eval(state);
}

void Expression::help() {
    printf("Variables:\n"
           "  x   \t the x coordinate, measured from 0 to width - 1\n"
           "  y   \t the y coordinate, measured from 0 to height - 1\n"
           "  t   \t the t coordinate, measured from 0 to frames - 1\n"
           "  c   \t the current channel\n"
           "  val \t the image value at the current x, y, t, c\n"
           "\n"
           "Constants:\n"
           "  e  \t 2.71828183\n"
           "  pi \t 3.14159265\n"
           "\n"
           "Uniforms:\n"
           "  frames \t the number of frames\n"
           "  width \t the image width\n"
           "  height \t the image height\n"
           "  channels \t the number of channels\n"
           "Unary Operations:\n"
           "  -  \t unary negation\n"
           "\n"
           "Binary Operations:\n"
           "  +  \t addition\n"
           "  -  \t subtraction\n"
           "  %%  \t modulo\n"
           "  *  \t multiplication\n"
           "  /  \t division\n"
           "  ^  \t exponentiation\n"
           "  >  \t greater than\n"
           "  <  \t less than\n"
           "  >= \t greater than or equal to\n"
           "  <= \t less than or equal to\n"
           "  == \t equal to\n"
           "  != \t not equal to\n"
           "\n"
           "Other Operations:\n"
           "  a ? b : c \t if a then b else c\n"
           "  [c]    \t sample the image here on channel c\n"
           "  [x, y] \t sample the image with a 3 lobed lanczos filter at X, Y at this channel\n"
           "  [x, y, t] \t sample the image with a 3 lobed lanczos filter at X, Y, T at this channel\n"
           "\n"
           "Functions:\n"
           "  log(x)      \t the natural log of x\n"
           "  exp(x)      \t e to the power of x\n"
           "  sin(x)      \t the sine of x\n"
           "  cos(x)      \t the cosine of x\n"
           "  tan(x)      \t the tangent of x\n"
           "  asin(x)     \t the inverse sine of x\n"
           "  acos(x)     \t the inverse cosine of x\n"
           "  atan(x)     \t the inverse tangent of x\n"
           "  atan2(y, x) \t the angle of the vector x, y above horizontal\n"
           "  abs(x)      \t the absolute value of x\n"
           "  floor(x)    \t the value of x rounded to the nearest smaller integer\n"
           "  ceil(x)     \t the value of x rounded to the nearest larger integer\n"
           "  round(x)    \t the value of x rounded to the nearest integer\n"
           "  mean()      \t the mean value of the image across all channels\n"
           "  mean(c)     \t the mean value of the image in channel c\n"
           "  sum()       \t the sum of the image across all channels\n"
           "  sum(c)      \t the sum of the image in channel c\n"
           "  max()       \t the maximum of the image across all channels\n"
           "  max(c)      \t the maximum of the image in channel c\n"
           "  min()       \t the minimum of the image across all channels\n"
           "  min(c)      \t the minimum of the image in channel c\n"
           "  stddev()    \t the standard deviation of the image across all channels\n"
           "  stddev(c)   \t the standard deviation of the image in channel c\n"
           "  variance()  \t the variance of the image across all channels\n"
           "  variance(c) \t the variance of the image in channel c\n"
           "  skew()      \t the skew of the image across all channels\n"
           "  skew(c)     \t the skew of the image in channel c\n"
           "  kurtosis()  \t the kurtosis of the image across all channels\n"
           "  kurtosis(c) \t the kurtosis of the image in channel c\n"
           "  covariance(c1, c2) \t the covariance between channels c1 and c2\n"
           "\n"
           "To add more functionality, see Parser.cpp in the source\n\n");
}





#include "footer.h"
