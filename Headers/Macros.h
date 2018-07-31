#pragma once

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
#include <Eigen/Eigen/Dense>
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

const int maxMoves = 121;
const int stateSize = 122;
