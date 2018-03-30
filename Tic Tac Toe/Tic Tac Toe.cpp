#include "Macros.h"
#include "FullyConnectedLayer.h"
#include "Neural Net.h"


pair<int, double> check(const Mat& tocheck, const Mat& correct)
{
	if (tocheck.rows() != correct.rows() || tocheck.cols() != correct.cols()) {
		printf("ERROR in check: Vectors are different sizes\n");
		return make_pair(0, 0);
	}
	int count = 0;
	double cost = 0.0;
	for (int col = 0; col < tocheck.cols(); col++)
	{
		bool works = true;
		for (int i = 0; i < tocheck.rows(); ++i) {
			double error = abs(tocheck(i, col) - correct(i, col));
			if (error >= 0.5)
				works = false;
			cost += error * error;
		}
		if (works)
			++count;
	}
	return make_pair(count, cost / tocheck.cols());
}

int main()
{
	srand(time(NULL));

	vector<Layer*> layers;
	layers.push_back(new SigLayer(10, 50));
	layers.push_back(new SigLayer(50, 10));
	Network2 n(layers, check, 10, 10, 16, 0.5);
    return 0;
}

