#include "../Headers/Macros.h"
#include "../Headers/Neural Net.h"
#include "../Headers/FullyConnectedLayer.h"
#include "../Headers/Node.h"

// Network2::Network2(const std::vector<Layer>& _layers, const checker_type& ch, int mbs, double lr, double maxr, double minr, double L2, double m)

// game-specific functions
void printBoardTTT(Vec s, bool ints) {
	if (s.rows() < 10) {
		cout << "Error: state size too small when printing\n";
		return;
	}
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (ints)
				printf("%d ", (int)s(3 * i + j));
			else
				printf("%.2lf ", s(3 * i + j));
		}
		printf("\n");
	}
	if (s.rows() == 10) {
		printf("Value: %.2lf\n", s(9));
	}
}

void printBoardC4(Vec s, bool isState) {
	if (isState) {
		if (s.rows() != stateSize) {
			cout << "Error: state size does not match when printing\n";
			return;
		}
		for (int i = 5; i >= 0; i--) {
			for (int j = 0; j < 7; j++) {
				if (s(7 * i + j) != -1)
					printf(" ");
				printf("%d ", (int)s(7 * i + j));
			}
			printf("\n");
		}
	}
	else {
		if (s.rows() != maxMoves + 1) {
			cout << "Error: probability distribution size does not match when printing\n";
			return;
		}
		for (int i = 0; i < 7; i++)
			printf("%.2lf ", s(i));
		printf("\nValue: %.2lf\n", s(7));
	}
	printf("\n");
}

Network2::Network2() {
	randGen = mt19937(randDev());
}

Network2::Network2(string _name) {
	ifstream fin;
	fin.open(_name + ".txt");
	fin >> *this;
	fin.close();
	name = _name;
	randGen = mt19937(randDev());
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
	if (this == &n)
		return *this;
	for (int i = 0; i < layers.size(); i++) {
		delete layers[i];
	}
	layers.clear();
	for (int i = 0; i < n.numLayers; i++)
		layers.push_back(n.layers[i]->copy());
	numLayers = n.numLayers;
	in = n.in; out = n.out;
	name = n.name;
	age = n.age;
	miniBatchSize = n.miniBatchSize;
	learnRate = n.learnRate;
	checker = n.checker; // is this neccessary?
	return *this;
}

Network2::~Network2() {
//	printf("In network2 destructor\n");
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
	if (input.rows() != out)
		cout << "Error: network output size is incorrect\n";
}

void Network2::simulate(Node* const start) const {
	Node* current = start;
	while (!current->isLeaf()) {
		current = current->chooseBest();
		if (current == NULL) {
			cout << "Error: current is nullptr in simulation\n";
			return; // end simulation early
		}
	}
	double v;
	if (!current->isEndState()) {
		Mat s = current->getState();
		feedForward(s);
		if (abs(s.sum() - s(s.rows() - 1, 0) - 1) > 0.0001) {
			cout << "Invalid probability distribution:\n";
			cout << s;
			return;
		}
		for (int i = 0; i < maxMoves+1; i++) {
			if (isnan(s(i))) {
				cout << "Error: Neural net output has nan! Output:\n";
				cout << s;
			}
		}
		v = s(maxMoves, 0);
		current->expand(s);
	}
	else {
		v = current->getEndVal();
	}
	while (current != start) {
		v = -v;
		(current->getParent())->update(v, current);
		current = current->getParent();
	}
}

int Network2::selectMove(const State& s, int moves, trbatch& data) const {
	Node* current = new Node(s, nullptr);
	for (int i = 0; i < moves; i++)
		simulate(current);
	int move = current->chooseMove();
	data.push_back(make_pair(s, current->getProbDistribution())); // value to be added later in fight
	delete current;
	return move;
}

void Network2::selfPlay(trbatch& trainingData) {
	const int simsPerMove = 100;
	State start = startState;
	Node* current = new Node(start, nullptr);
	while (!current->isEndState()) {
		for (int i = 0; i < simsPerMove; i++)
			simulate(current);
		current = current->chooseNewState();
	}
	double winner = current->getEndVal();
	Node* prev = current;
	current = current->getParent();
	while (current != nullptr) {
		trdata data = make_pair(current->getState(), current->getProbDistribution());
		winner = -winner;
		data.second(maxMoves) = winner;
		trainingData.push_back(data);
		prev = current;
		current = current->getParent();
	}
	delete prev;
}

const Network2& fight(const Network2& n1, const Network2& n2, trbatch& data) {
	cout << "Fight Result: ";
	const int simsPerFightMove = 100;
	const int games = 50;
	int win1 = 0, win2 = 0, lastInd = data.size(); // lastInd is the last index in data that needs the result to be added
	for (int i = 0; i < games; i++) {
		State state = startState;
		int turn = 2 * (i % 2) - 1;
		while (true) {
			int move;
			if (turn == 1)
				move = n1.selectMove(state, simsPerFightMove, data);
			else
				move = n2.selectMove(state, simsPerFightMove, data);
			state = Node::nextStateC4(state, move);
			auto pair = Node::evaluateStateC4(state);
			if (pair.first) {
				double winner; // the winner of the last state, from that state's perspective
				if (pair.second == 1) {
					if (turn == 1)
						win1++;
					else
						win2++;
					winner = 1;
				}
				else if (pair.second == -1) {
					cout << "Error: Impossible to win on opponent's turn\n";
				}
				else
					winner = 0;
				for (int i = data.size() - 1; i >= lastInd; i--) {
					data[i].second(maxMoves) = winner;
					winner = -winner;
				}
				lastInd = data.size();
				break;
			}
			state = -state;
			turn = -turn;
		}
		if (i % 10 == 9) {
			cout << win1 << " to " << win2 << "\n";			
		}
	}
	cout << win1 << " to " << win2 << "\n";
	if (win1 > win2)
		return n1;
	else
		return n2;
}

void Network2::learn(trbatch& data) {
	bool showSample = true;
	shuffle(data.begin(), data.end(), randGen);
	Mat batch(in, miniBatchSize);
	Mat answers(out, miniBatchSize);
	for (int i = 0; i < data.size(); ++i) {
		batch.col(i % miniBatchSize) = data[i].first;
		answers.col(i % miniBatchSize) = data[i].second;
		if ((i + 1) % miniBatchSize == 0) {
			if (showSample && i + 1 == miniBatchSize) {
				cout << "\nSample state:\n";
				printBoardC4(batch.col(0), true);
			}
			feedForward(batch);
			if (showSample && i + 1 == miniBatchSize) {
				cout << "Prediction:\n";
				printBoardC4(batch.col(0), false);
				cout << "Answers:\n";
				printBoardC4(answers.col(0), false);
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
}

void Network2::train(int iterations) {
	const int gamesPerIteration = 100;
	ofstream fout;
	Network2 before;
//	Network2 best(name + " Best");
	Network2 roleModel("Connect4 2 (200-200-200)");
	for (int i = 0; i < iterations; i++) {
		trbatch data;
		before = *this; // the state of the network before this iteration

		// generate data from self-play games
		for (int game = 0; game < gamesPerIteration; game++)
			selfPlay(data);

		learn(data);

		// check whether neural net has improved during the iteration
		cout << "Before vs Current\n";
		fight(before, *this, data);

		/*
		cout << "\nBest vs Current\n";
		best = fight(best, *this, data);

		fout.open(name + " best.txt");
		fout << best;
		fout.close();
		*/
		cout << "\nRole Model vs Current\n";
		fight(roleModel, *this, data);

		// use data from fight
		learn(data);

		age++;
		cout << "Age: " << age << "\n";

		fout.open(name + ".txt");
		fout << *this;
		fout.close();

		if (age % 100 == 0) { // save record of progress every 100 iterations
			fout.open(name + " at iteration " + to_string(age) + ".txt");
			fout << *this;
			fout.close();
		}
		
		cout << "Saved\n====================================\n";
	}
}

void Network2::play() {
	const int simsPerGameMove = 5000;
	trbatch data;
	int keepPlaying, first;
	do {
		cout << "Do you want to go first? (Enter 0 or 1) ";
		cin >> first;
		int turn = 2 * first - 1, move;
		State state = startState;
		while (true) {
			if (turn == 1) { // Human Turn
				printBoardC4(state, true);
				cout << "Your Move: ";
				vector<bool> valid = Node::validMovesC4(state);
				cin >> move;
				while (move < 0 || move >= 7 || !valid[move]) {
					cout << "Invalid Move! Choose a new move: ";
					cin >> move;
				}
				state = Node::nextStateC4(state, move);
			}
			else { // Computer Turn
				move = selectMove(state, simsPerGameMove, data);
				state = Node::nextStateC4(state, move);
				cout << "Computer's Move: " << move << "\n";
			}
			auto pair = Node::evaluateStateC4(state);
			if (pair.first) {
				if (pair.second == 1) {
					if (turn == 1)
						cout << "You won\n";
					else
						cout << "Computer won\n";
				}
				else if (pair.second == -1)
					cout << "Error: impossible to win on opponent's turn\n";
				else
					cout << "Tie\n";
				printBoardC4(turn*state, true); // show final state from your point of view
				break;
			}
			state = -state;
			turn = -turn;
		}
		cout << "Do you want to play again? (Enter 0 or 1) ";
		cin >> keepPlaying;
	} while (keepPlaying);
}
