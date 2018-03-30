#include "Macros.h"
#include "Neural Net.h"

using namespace std;

// Network2::Network2(const std::vector<Layer>& _layers, const checker_type& ch, int mbs, double lr, double maxr, double minr, double L2, double m)

Network2::Network2(const std::vector<Layer*>& _layers, const checker_type& ch, int _in, int _out, int mbs, double lr) {
	layers = _layers;
	checker = ch;
	miniBatchSize = mbs;
	learnRate = lr;
	//	maxRate = maxr;
	//	minRate = minr;
	// L2 = _L2;
	//	momentum = m;
	numLayers = layers.size();
	randGen = mt19937(randDev());
	in = _in;
	out = _out;
}

Network2::~Network2() {
	printf("In network2 destructor\n");
	for (Layer* l : layers) {
		if (l != NULL) delete l;
		l = NULL;
	}
}

void Network2::feedForward(Mat& input) {
	for (int i = 0; i < numLayers; ++i)
		layers[i]->apply(input);
}

// move to new file
class Node
{
private:
	Node * children[9];
	Node* parent;
	Vec state;
	bool valid[9]; // whether a move is valid
	int N[9]; // visit count
	double W[9]; // total action value
	double Q[9]; // mean action value
	double P[9]; // prior probability
	bool leaf;
	const double c_puct = 0.01; // change?

public:
	Node(Vec s, Node* p) {
		if (s.rows != 10)
			cout << "Error: node initialize with invalid state\n";
		state = s;
		parent = p;
		for (int i = 0; i < 9; i++) {
			valid[i] = (state[i] == 0);
			N[i] = 0;
			W[i] = 0;
			Q[i] = 0;
			P[i] = 0;
		}
		leaf = true;
	}
	Node* chooseBest() {
		if (leaf) {
			cout << "Error: choosing from leaf node\n";
			return nullptr;
		}
		double totalVisits = 0;
		for (int i = 0; i < 9; i++)
			totalVisits += N[i];
		double bestVal = -1; int bestMove = -1;
		for (int i = 0; i < 9; i++) {
			if (valid[i]) {
				double val = Q[i] + c_puct * P[i] * sqrt(totalVisits) / (1 + N[i]);
				if (val > bestVal) {
					bestVal = val;
					bestMove = i;
				}
			}
		}
		return children[bestMove];
	}
	void expand() {
		if (!leaf) {
			cout << "Error: tried to expand non-leaf node\n";
			return;
		}
		leaf = false;
		// add call to network for probabilities
		for (int i = 0; i < 9; i++) {
			if (valid[i]) {
				Vec copy = state;
				copy[i] = state[9];
				copy[9] = -copy[9];
				children[i] = new Node(copy, this);
			}
			else
				children[i] = nullptr;
		}
	}
	void destroy() { // to deallocate memory
		for (int i = 0; i < 9; i++) {
			children[i]->destroy();
			delete children[i];
		}
	}
};

Mat play() {
	Vec start = Vec::Zero(10);
	start[9] = 1;
	Node current = Node(start, nullptr);
}

void Network2::train(trbatch& data, trbatch& test, int numEpochs) {
	for (int epoch = 1; epoch <= numEpochs; ++epoch)
	{
		shuffle(data.begin(), data.end(), randGen);
		Mat batch(in, miniBatchSize);
		Mat answers(out, miniBatchSize);
		for (int i = 0; i < data.size(); ++i) {
			batch.col(i % miniBatchSize) = data[i].first;
			answers.col(i % miniBatchSize) = data[i].second;
			if ((i + 1) % miniBatchSize == 0) {
				// feedforward
				feedForward(batch);

				// backpropagate error
				Mat WTD;
				layers[numLayers - 1]->computeDeltaLast(batch, answers, WTD);
				for (int i = numLayers - 2; i >= 0; i--) {
					layers[i]->computeDeltaBack(WTD);
				}
				// updates
				for (int i = 0; i < numLayers; i++) {

					layers[i]->updateBiasAndWeights(learnRate);
				}
				batch.resize(in, miniBatchSize);
				answers.resize(out, miniBatchSize);
			}
		}
		// evaluate progress
		Mat testBatch(in, test.size());
		Mat testAns(out, test.size());
		for (int i = 0; i < test.size(); ++i) {
			testBatch.col(i) = test[i].first;
			testAns.col(i) = test[i].second;
		}
		feedForward(testBatch);

		if (isnan(testBatch(0, 0)))
		{
			cout << "NAN!\n";
			layers[1]->print();
		}

		auto p = checker(testBatch, testAns);
		printf("Epoch %d: %d out of %lu correct, average cost: %.3f\n", epoch, p.first, testBatch.cols(), p.second);
	}
}
