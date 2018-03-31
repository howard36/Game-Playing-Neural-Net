#pragma once

#include <Eigen/Dense>
#include "Macros.h"

// The layer interface. Is an abstract class

typedef Eigen::MatrixXd Mat;

class Layer {
public:

	Layer();
	virtual ~Layer()=0;

	virtual void apply(Mat& input)=0;

	// if this layer is the last layer, computes the delta (error) given the output and correct answer
	virtual void computeDeltaLast(const Mat& output, const Mat& ans, Mat& WTD)=0;

	// if this layer is not the last layer, computes the delta from the last layer's delta

	virtual void computeDeltaBack(Mat& WTD)=0;

	virtual void updateBiasAndWeights(double lrate)=0;

	virtual std::pair<int, int> getSize()=0;

	virtual void print()=0;
};