#pragma once

#include "Macros.h"

struct SigmoidActivationFunction {
    static Mat activation(const Mat &a);
    static Mat activationDeriv(const Mat &x);
    static int id();
};

struct TanhActivationFunction {
    static Mat activation(const Mat &a);
    static Mat activationDeriv(const Mat &x);
    static int id();
};

struct SoftMaxActivationFunction {
    static Mat activation(const Mat &a);
    static Mat activationDeriv(const Mat &x);
    static int id();
};

struct CustomActivationFunction {
    static Mat activation(const Mat &a);
    static Mat activationDeriv(const Mat &x);
    static int id();
};
