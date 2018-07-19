#include "../Headers/Macros.h"
#include "../Headers/FullyConnectedLayer.h"
#include "../Headers/ActivationFunction.h"
#include "../Headers/Neural Net.h"

typedef FullyConnectedLayer<SigmoidActivationFunction> SigmoidLayer;
typedef FullyConnectedLayer<SoftMaxActivationFunction> SoftMaxLayer;
typedef FullyConnectedLayer<CustomActivationFunction> CustomLayer;
/*
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
		double colCost = 0;
		int max = 0;
		for (int i = 0; i < tocheck.rows(); ++i) {
//			double error = abs(tocheck(i, col) - correct(i, col));
//			if (error >= 0.5)
//				works = false;
//			cost += error * error;
			colCost += -correct(i, col)*log(tocheck(i, col));
			if (tocheck(i, col) > tocheck(max, col))
				max = i;
		}
		if (correct(max, col))
			++count;
		cost += colCost;
	}
	return make_pair(count, cost / tocheck.cols());
}
*/
int main()
{
	srand(time(NULL));
	/*
	vector<Layer*> layers;
	layers.push_back(new SigmoidLayer(stateSize, 500));
	layers.push_back(new SigmoidLayer(500, 500));
	layers.push_back(new CustomLayer(500, maxMoves+1));
	Network2 n("Test", layers, check, stateSize, maxMoves+1, 16, 0.1);
	*/

	trbatch data;

	Network2 n("Connect4 3 (500-500) branch 5.2");
	Network2 n2("Connect4 3 (500-500) branch 5.1 at iteration 801");
	// fight(n, n2, 1000, 200, data);

	// n.play();
	printf("Start Training\n");
	n.train(50, 50);

	cout << "Done!\n";
    return 0;
}
