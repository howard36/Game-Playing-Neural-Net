#include "Node.h"
#include "Macros.h"

vector<bool> validMovesTTT(State s) {
	vector<bool> v(maxMoves);
	for (int i = 0; i < maxMoves; i++) {
		v[i] = (s(i) == 0);
	}
	return v;
}

vector<bool> validMovesC4(State s) {
	vector<bool> v(maxMoves);
	for (int i = 0; i < maxMoves; i++) {
		v[i] = (s(i + 35) == 0);
	}
	return v;
}

int choose(Vec distribution) {
	if (distribution.size() != maxMoves+1) {
		cout << "Error: distribution size does not match\n";
	}
	if (abs(distribution.sum() - distribution(maxMoves) - 1) > 0.0001) {
		cout << "Error: probabilities do not add up to 1\n";
		cout << distribution;
	}
	double num = (double)rand() / RAND_MAX;
	for (int i = 0; i < maxMoves; i++) {
		if (num < distribution(i)) {
			return i;
		}
		num -= distribution(i);
	}
	// if no random move has been chosen yet (which happens rarely), it chooses the one with greatest probability
	double bestProb = 0; int bestMove = -1;
	for (int i = 0; i < maxMoves; i++) {
		if (distribution(i) > bestProb) {
			bestProb = distribution(i);
			bestMove = i;
		}
	}
	return bestMove;
}

// decides whether the state is an end state, and if so, who the winner is
// specialized for the game of tic tac toe
pair<bool, double> Node::evaluateStateTTT(State s) {
	// check for 3 in a rows
	for (int i = 0; i < 3; i++) {
		if (s(3 * i) == s(3 * i + 1) && s(3 * i + 1) == s(3 * i + 2)) {
			if (s(3 * i) != 0)
				return make_pair(true, s(3 * i));
		}
		if (s(i) == s(i + 3) && s(i + 3) == s(i + 6)) {
			if (s(i) != 0)
				return make_pair(true, s(i));
		}
	}
	if (s(0) == s(4) && s(4) == s(8)) {
		if (s(0) != 0)
			return make_pair(true, s(0));
	}
	if (s(2) == s(4) && s(4) == s(6)) {
		if (s(2) != 0)
			return make_pair(true, s(2));
	}
	// no 3 in a rows found
	for (int i = 0; i < 9; i++) {
		if (s(i) == 0) // available move
			return make_pair(false, 0);
	}
	return make_pair(true, 0); // tie
}

// decides whether the state is an end state, and if so, who the winner is
// specialized for the game of connect 4
pair<bool, double> Node::evaluateStateC4(State s) {
	// horizontal
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j <= 3; j++) {
			if (s(7 * i + j) == s(7 * i + j + 1) && s(7 * i + j + 1) == s(7 * i + j + 2) && s(7 * i + j + 2) == s(7 * i + j + 3) && s(7 * i + j) != 0) {
				return make_pair(true, s(7 * i + j));
			}
		}
	}
	// vertical
	for (int i = 0; i <= 2; i++) {
		for (int j = 0; j < 7; j++) {
			if (s(7 * i + j) == s(7 * i + j + 7) && s(7 * i + j + 7) == s(7 * i + j + 14) && s(7 * i + j + 14) == s(7 * i + j + 21) && s(7 * i + j) != 0) {
				return make_pair(true, s(7 * i + j));
			}
		}
	}
	// diagonal up-right
	for (int i = 0; i <= 2; i++) {
		for (int j = 0; j <= 3; j++) {
			if (s(7 * i + j) == s(7 * i + j + 8) && s(7 * i + j + 8) == s(7 * i + j + 16) && s(7 * i + j + 16) == s(7 * i + j + 24) && s(7 * i + j) != 0) {
				return make_pair(true, s(7 * i + j));
			}
		}
	}
	// diagonal up-left
	for (int i = 0; i <= 2; i++) {
		for (int j = 3; j < 7; j++) {
			if (s(7 * i + j) == s(7 * i + j + 6) && s(7 * i + j + 6) == s(7 * i + j + 12) && s(7 * i + j + 12) == s(7 * i + j + 18) && s(7 * i + j) != 0) {
				return make_pair(true, s(7 * i + j));
			}
		}
	}
	// there are still moves available
	for (int i = 0; i < 7; i++) {
		if (s(i + 35) == 0)
			return make_pair(false, 0);
	}
	// tie
	return make_pair(true, 0);
}

State Node::nextStateTTT(State s, int move) {
	State copy = s;
	if (copy(move) != 0)
		cout << "Error: Invalid move\n";
	copy(move) = 1;
	return copy;
}

State Node::nextStateC4(State s, int move) {
	State copy = s;
	if (copy(move + 35) != 0)
		cout << "Error: Invalid move\n";
	for (int i = move; i < 42; i += 7) { // move up the column looking for empty space
		if (copy(i) == 0) {
			copy(i) = 1;
			break;
		}
	}
	return copy;
}

Node::Node(State s, Node* p) {
	if (s.rows() != stateSize) {
		cout << "Error: node initialized with invalid state\n";
	}
	state = s;
	parent = p;
	valid = validMovesC4(state);
	auto pair = evaluateStateC4(state);
	end = pair.first;
	endVal = pair.second;
	leaf = true;
	for (int i = 0; i < maxMoves; i++) {
		children[i] = nullptr;
		N[i] = 0;
		W[i] = 0;
		Q[i] = 0;
		P[i] = 0;
	}
}

Node::~Node() {
	for (int i = 0; i < maxMoves; i++) {
		if (children[i] != NULL) {
			delete children[i];
		}
	}
}

// used for MCTS simulations
Node* Node::chooseBest() {
	if (leaf) {
		cout << "Error: choosing from leaf node\n";
		return nullptr;
	}
	double totalVisits = 0;
	for (int i = 0; i < maxMoves; i++)
		totalVisits += N[i];
	double bestVal = -2; int bestMove = -1;
	for (int i = 0; i < maxMoves; i++) {
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

void Node::expand(Vec prob) { // expands node
	if (!leaf) {
		cout << "Error: tried to expand non-leaf node\n";
		return;
	}
	if (prob.rows() != maxMoves+1) { // the last element is the predicted state value, not part of the probability distribution
		cout << "Error: probability vector size does not match\n";
		cout << "Probability size is " << prob.rows() << "\n";
		return;
	}
	leaf = false;
	for (int i = 0; i < maxMoves; i++) {
		if (valid[i]) {
			State copy = nextStateC4(state, i);
			copy = -copy; // switch to opponent's point of view
			children[i] = new Node(copy, this);
			P[i] = prob(i);
		}
		else {
			children[i] = nullptr;
		}
	}
}

void Node::update(double v, Node* child) {
	for (int i = 0; i < maxMoves; i++) {
		if (children[i] == child) {
			N[i]++;
			W[i] += v;
			Q[i] = W[i] / N[i];
		}
	}
}

Vec Node::getProbDistribution() {
	Vec v(maxMoves+1); // last element is reserved for game winner, which is added later
	double sum = 0;
	for (int i = 0; i < maxMoves; i++) {
		sum += N[i];
	}
	for (int i = 0; i < maxMoves; i++) {
		v(i) = N[i] / sum;
	}
	v(maxMoves) = 0;
	return v;
}

// used for game after simulations are complete
// based on visit counts
int Node::chooseMove() {
	int next = choose(getProbDistribution());
	if (children[next] == NULL) {
		cout << "Error: the move chosen is null\nVisit Counts: ";
		for (int i = 0; i < maxMoves; i++)
			printf("%d ", N[i]);
		printf("\nMove Chosen: %d\n", next);
	}
	return next;
}

Node* Node::chooseNewState() {
	if (leaf)
		cout << "Error: choosing move from leaf node\n";
	return children[chooseMove()];
}