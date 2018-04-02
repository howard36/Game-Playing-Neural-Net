#include "Macros.h"
#include "Neural Net.h"
#include "FullyConnectedLayer.h"

using namespace std;

// Network2::Network2(const std::vector<Layer>& _layers, const checker_type& ch, int mbs, double lr, double maxr, double minr, double L2, double m)

Network2::Network2() {
}

Network2::Network2(string _name) {
	ifstream fin;
	fin.open(_name + ".txt");
	fin >> *this;
	fin.close();
	name = _name;
}

Network2::Network2(const string _name, const std::vector<Layer*>& _layers, const checker_type& ch, int _in, int _out, int mbs, double lr) {
	name = _name;
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

Network2::Network2(Network2& n) {
	*this = n;
}

Network2& Network2::operator= (const Network2& n) {
	ofstream fout; ifstream fin;
	fout.open("Network Exchange.txt");
	fout << n;
	fout.close();
	fin.open("Network Exchange.txt");
	fin >> *this;
	fin.close();
	return *this;
}

Network2::~Network2() {
	printf("In network2 destructor\n");
	for (Layer* l : layers) {
		if (l != NULL) delete l;
		l = NULL;
	}
}

ifstream& operator>> (ifstream& fin, Network2& n) {
	for (int i = 0; i < n.layers.size(); i++) {
		delete n.layers[i];
	}
	n.layers.clear();
	fin >> n.numLayers >> n.miniBatchSize >> n.learnRate >> n.age;
	for (int i = 0; i < n.numLayers; i++) {
		n.layers.push_back(read(fin));
		if (n.layers[i] == NULL) {
			cout << "Error: Network has null layer\n";
		}
	}
	n.in = n.layers[0]->getSize().first;
	n.out = n.layers[n.numLayers - 1]->getSize().second;
	return fin;
}

ofstream& operator<< (ofstream& fout, const Network2& n) {
	fout << n.numLayers << " " << n.miniBatchSize << " " << n.learnRate << " " << n.age << "\n";
	for (int i = 0; i < n.numLayers; i++) {
		n.layers[i]->write(fout);
	}
	return fout;
}

void Network2::setChecker(const checker_type& ch) { checker = ch; }

void Network2::feedForward(Mat& input) const {
	for (int i = 0; i < numLayers; ++i)
		layers[i]->apply(input);
}

vector<bool> validMoves(Vec state) {
	vector<bool> v(9, false);
	for (int i = 0; i < 9; i++) {
		if (state(i) == 0) {
			v[i] = true;
		}
	}
	return v;
}

int choose(Vec distribution) {
	if (distribution.size() != 10) {
		cout << "Error: distribution size does not match\n";
	}
	if (abs(distribution.sum() - distribution(9) - 1) > 0.0001) {
		cout << "Error: probabilities do not add up to 1\n";
		cout << distribution;
	}
	double num = (double)rand() / RAND_MAX;
	for (int i = 0; i < 9; i++) {
		if (num < distribution(i)) {
			return i;
		}
		num -= distribution(i);
	}
	// if no random move has been chosen yet (which happens rarely), it chooses the one with greatest probability
	double bestProb = 0; int bestMove = -1;
	for (int i = 0; i < 9; i++) {
		if (distribution(i) > bestProb) {
			bestProb = distribution(i);
			bestMove = i;
		}
	}
	return bestMove;
}

// decides whether the state is an end state, and if so, who the winner is
// specialized for the game of tic tac toe
pair<bool, double> evaluateState(Vec state) {
	// check for 3 in a rows
	for (int i = 0; i < 3; i++) {
		if (state(3 * i) == state(3 * i + 1) && state(3 * i + 1) == state(3 * i + 2)) {
			if (state(3 * i) != 0)
				return make_pair(true, state(3 * i));
		}
		if (state(i) == state(i + 3) && state(i + 3) == state(i + 6)) {
			if (state(i) != 0)
				return make_pair(true, state(i));
		}
	}
	if (state(0) == state(4) && state(4) == state(8)) {
		if (state(0) != 0)
			return make_pair(true, state(0));
	}
	if (state(2) == state(4) && state(4) == state(6)) {
		if (state(2) != 0)
			return make_pair(true, state(2));
	}
	// no 3 in a rows found
	for (int i = 0; i < 9; i++) {
		if (state(i) == 0) // available move
			return make_pair(false, 0);
	}
	return make_pair(true, 0); // tie
}

// move to new file
class Node
{
private:
	Node* children[9];
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
		if (s.rows() != 9)
			cout << "Error: node initialized with invalid state\n";
		state = s;
		parent = p;
		valid = validMoves(state);
		auto pair = evaluateState(state);
		end = pair.first;
		endVal = pair.second;
		leaf = true;
		for (int i = 0; i < 9; i++) {
			N[i] = 0;
			W[i] = 0;
			Q[i] = 0;
			P[i] = 0;
		}

	}
	~Node() {
		for (int i = 0; i < 9; i++) {
			if (children[i] != NULL) {
				delete children[i];
			}
		}
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
		double bestVal = -2; int bestMove = -1;
		for (int i = 0; i < 9; i++) {
			if (valid[i]) {
				double val = Q[i] + c_puct * P[i] * sqrt(totalVisits) / (1 + N[i]);
				if (val > bestVal) {
					bestVal = val;
					bestMove = i;
				}
			}
		}
		if (bestMove == -1) {
			cout << "Error: No best move found\n"; // Probabilities might be nan!
			return nullptr;
		}
		return children[bestMove];
	}
	void expand(Vec prob) { // expands node
		if (!leaf) {
			cout << "Error: tried to expand non-leaf node\n";
			return;
		}
		if (prob.rows() != 10) { // the last element is the predicted state value, not part of the probability distribution
			cout << "Error: probability vector size does not match\n";
			return;
		}
		leaf = false;
		for (int i = 0; i < 9; i++) {
			if (valid[i]) {
				Vec copy = state;
				copy(i) = 1;
				for (int j = 0; j < 9; j++)
					copy(j) = -copy(j); // show next state from opponent's perspective
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
		v(9) = 0;
		return v;
	}
	// used for game after simulations are complete
	// based on visit counts
	int chooseMove() {
		int next = choose(getProbDistribution());
		if (children[next] == NULL) {
			cout << "Error: the move chosen is null\nVisit Counts: ";
			for (int i = 0; i < 9; i++)
				printf("%d ", N[i]);
			printf("\nMove Chosen: %d\n", next);
		}
		return next;
	}
	Node* chooseNewState() {
		if (leaf)
			cout << "Error: choosing move from leaf node\n";
		return children[chooseMove()];
	}
	void printDist() {
		printf("Distribution: ");
		for (int i = 0; i < 9; i++)
			printf("%d ", N[i]);
		printf("\nPrior Probabilities: ");
		for (int i = 0; i < 9; i++)
			printf("%lf ", P[i]);
		printf("\nState: ");
		for (int i = 0; i < 9; i++)
			printf("%lf ", state(i));
		printf("\nValid Moves: ");
		for (int i = 0; i < 9; i++)
			cout << valid[i] << " ";
		printf("\n");
	}
};

void Network2::simulate(Node * const start) const {
	Node* current = start;
	while (current != nullptr && !current->isLeaf())
		current = current->chooseBest();
	double v;
	if (current != nullptr && !current->isEndState()) {
		Mat s = current->getState();
		feedForward(s);
		if (abs(s.sum() - s(s.rows() - 1) - 1) > 0.0001) {
			cout << "Invalid probability distribution\n";
			cout << s;
			return;
		}
		if (isnan(s(0)))
			cout << "NAN in simulation!\n";
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

int Network2::selectMove(const Vec& state, int moves) const {
	Vec start = state;
	Node* current = new Node(start, nullptr);
	for (int i = 0; i < moves; i++)
		simulate(current);
	return current->chooseMove();
}

void Network2::selfPlay(trbatch& trainingData) {
	const int simulationsPerMove = 1000;
	Vec start = Vec::Zero(9);
	Node* current = new Node(start, nullptr);
	while (!current->isEndState()) {
		for (int i = 0; i < simulationsPerMove; i++)
			simulate(current);
		current = current->chooseNewState();
	}
	int winner = current->getEndVal();
	Node* prev = current;
	current = current->getParent();
	while (current != nullptr) {
		trdata data = make_pair(current->getState(), current->getProbDistribution());
		data.second(9) = winner;
		trainingData.push_back(data);
		prev = current;
		current = current->getParent();
	}
	delete prev;
}

const Network2& fight(const Network2& n1, const Network2& n2) {
	const int simulationsPerFightMove = 10;
	const int games = 100;
	int win1 = 0, win2 = 0;
	for (int i = 0; i < games; i++) {
		Vec state = Vec::Zero(9);
		int turn = 2 * (i % 2) - 1;
		while (true) {
			if (turn == 1) {
				state(n1.selectMove(state, simulationsPerFightMove)) = turn;
			}
			else {
				state(n2.selectMove(-state, simulationsPerFightMove)) = turn;
			}
			auto pair = evaluateState(state);
			if (pair.first) {
				if (pair.second != 0) {
					if (pair.second*turn < 0) {
						cout << "Error: Impossible for to win game on opponent's turn\n";
						break;
					}
					if (turn == 1)
						win1++;
					else
						win2++;
				}
				break;
			}
			turn = -turn;
		}
	}
	cout << "Network Fight: " << win1 << " to " << win2 << "\n";
	if (win1 > win2) {
		return n1;
	}
	else {
		return n2;
	}
}

void printBoard(Vec s) {
	if (s.rows() < 9) {
		cout << "Error: state size too small when printing\n";
		return;
	}
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			printf("%.2lf ", s(3 * i + j));
		}
		printf("\n");
	}
	if (s.rows() == 10) {
		printf("Value: %.2lf\n", s(9));
	}
}

void Network2::train(int iterations) {
	ofstream fout;
	const int gamesPerIteration = 100;
	for (int i = 0; i < iterations; i++)
	{
		Network2 before(*this); // the state of the network before this iteration
		trbatch data;
		for (int game = 0; game < gamesPerIteration; game++) {
			selfPlay(data);
		}
		shuffle(data.begin(), data.end(), randGen);
		Mat batch(in, miniBatchSize);
		Mat answers(out, miniBatchSize);
		for (int i = 0; i < data.size(); ++i) {
			batch.col(i % miniBatchSize) = data[i].first;
			answers.col(i % miniBatchSize) = data[i].second;
			if ((i + 1) % miniBatchSize == 0) {
				if (i + 1 == miniBatchSize) {
					cout << "\nSample state:\n";
					printBoard(batch.col(0));
				}
				feedForward(batch);
				if (i + 1 == miniBatchSize) {
					cout << "\nPrediction:\n";
					printBoard(batch.col(0));
					cout << "\nAnswers:\n";
					printBoard(answers.col(0));
				}
				if (isnan(batch(0, 0)))
					cout << "NAN!\n";

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

		// check whether neural net has improved during the iteration
		*this = fight(*this, before);

		fout.open(name + ".txt");
		fout << *this;
		fout.close();
		if (age % 100 == 0) { // save record of progress every 100 iterations
			fout.open(name+" at iteration " + to_string(age) + ".txt");
			fout << *this;
			fout.close();
		}

		age++;
		printf("\nAge: %d\n", age);
	}
}