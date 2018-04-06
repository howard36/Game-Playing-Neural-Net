#pragma once

//#include "FullyConnectedLayer.h"

#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <string>
#include <fstream>
#include <functional>
#include <chrono>
#include <Eigen/Dense>
using namespace std;

typedef long long ll;

typedef std::vector<double> vdbl;
typedef std::vector<vdbl> v2dbl;
typedef std::vector<v2dbl> v3dbl;

typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vec;
typedef Vec State;

typedef std::pair<Vec, Vec> trdata;
typedef std::vector<trdata> trbatch;

typedef std::function<std::pair<int, double>(const Mat&, const Mat&) > checker_type;

const int maxMoves = 7;
const int stateSize = 42;
const State startState = Vec::Zero(stateSize);