#pragma once

#include "Macros.h"

struct SigmoidActivationFunction {
	static Mat activation(const Mat& a);
	static Mat activationDeriv(const Mat& x);
};

struct TanhActivationFunction {
	static Mat activation(const Mat& a);
	static Mat activationDeriv(const Mat& x);
};

struct SoftMaxActivationFunction {
	static Mat activation(const Mat& a);
	static Mat activationDeriv(const Mat& x);
};