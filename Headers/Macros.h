#pragma once

#include <eigen3/Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>
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

const int maxMoves = 7; // game dependent
const int stateSize = 42; // game dependent
