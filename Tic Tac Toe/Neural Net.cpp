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

vector<bool> validMoves(Vec state) {
	vector<bool> v(9, false);
	for (int i = 0; i < 9; i++)
		v[i] = (state(i) == 0);
	return v;
}

int choose(Vec distribution) {
	double num = (double)rand() / RAND_MAX;
	for (int i = 0; i < 9; i++) {
		if (num <= distribution(i))
			return i;
		num -= distribution(i);
	}
	cout << "Error: no random move was not chosen\n";
	return 0;
}

// decides whether the state is an end state, and if so, who the winner is
pair<bool, int> evaluateState(Vec state) {
	return make_pair(false, 0);
}

// move to new file
class Node
{
private:
	Node * children[9];
	Node* parent;
	Vec state;
	vector<bool> valid; // whether a move is valid
	int N[9]; // visit count
	double W[9]; // total action value
	double Q[9]; // mean action value
	double P[9]; // prior probability
	bool leaf; // whether this is currently a leaf node in the game tree
	bool end; // whether this is an ending state
	double endVal; // the winner (if it is an end state)
	const double c_puct = 0.01; // change?

public:
	Node(Vec s, Node* p) {
		if (s.rows() != 10)
			cout << "Error: node initialize with invalid state\n";
		state = s;
		parent = p;
		valid = validMoves(state);
		auto pair = evaluateState(state);
		end = pair.first;
		endVal = pair.second;
		for (int i = 0; i < 9; i++) {
			N[i] = 0;
			W[i] = 0;
			Q[i] = 0;
			P[i] = 0;
		}
		leaf = true;
	}
	// used for MCTS simulations
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
	void expand(Vec prob) { // expands node, returns evaluation
		if (!leaf) {
			cout << "Error: tried to expand non-leaf node\n";
			return;
		}
		if (prob.rows() != 10) {
			cout << "Error: probability vector size does not match\n";
			return;
		}
		leaf = false;
		for (int i = 0; i < 9; i++) {
			if (valid[i]) {
				Vec copy = state;
				copy[i] = state[9];
				copy[9] = -copy[9];
				children[i] = new Node(copy, this);
				P[i] = prob(i, 0);
			}
			else
				children[i] = nullptr;
		}
	}
	void update(double v, Node* child) {
		for (int i = 0; i < 9; i++) {
			if (children[i] == child) {
				N[i]++;
				W[i] += v;
				Q[i] = W[i] / N[i];
			}
		}
	}
	void destroy() { // to deallocate memory
		for (int i = 0; i < 9; i++) {
			children[i]->destroy();
			delete children[i];
		}
	}
	bool isLeaf() { return leaf; }
	Node* getParent() { return parent; }
	Vec getState() { return state; }
	bool isEndState() { return end; }
	double getEndVal() { return endVal; }
	Vec getProbDistribution() {
		Vec v(10); // last element is reserved for game winner, which is added later
		double sum = 0;
		for (int i = 0; i < 9; i++) {
			sum += N[i];
		}
		for (int i = 0; i < 9; i++) {
			v(i) = N[i] / sum;
		}
		return v;
	}
	// used for game after simulations are complete
	// based on visit counts
	Node* chooseMove() {
		int next = choose(getProbDistribution());
		// discard the rest of the game tree
		for (int i = 0; i < 9; i++) {
			if (i != next)
				children[i]->destroy();
		}
		return children[next];
	}
};

void Network2::simulate(Node* start) {
	Node* current = start;
	while (!current->isLeaf())
		current = current->chooseBest();
	double v;
	if (!current->isEndState()) {
		Mat s = current->getState();
		feedForward(s);
		v = s(9, 0);
		current->expand(s);
	}
	else {
		v = current->getEndVal();
	}
	while (current != start) {
		(current->getParent())->update(v, current);
		current = current->getParent();
	}
}

Vec Network2::selfPlay() {
	const int simulationsPerMove = 1000;
	vector<Vec> trainingData;
	Vec start = Vec::Zero(10);
	start[9] = 1;
	Node* current = new Node(start, nullptr);
	while (!current->isEndState()) {
		for (int i = 0; i < simulationsPerMove; i++)
			simulate(current);
		current = current->chooseMove();
	}
	int winner = current->getEndVal();
	while (current != nullptr) {
		Vec data = current->getProbDistribution();
		data(10) = winner;
		trainingData.push_back(data);
		current = current->getParent();
	}
	int random = rand() % trainingData.size();
	return trainingData[random];
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
