#include "../Headers/Node.h"
#include "../Headers/Macros.h"

vector<bool> Node::validMovesTTT(State s) {
    vector<bool> v(maxMoves);
    for (int i = 0; i < maxMoves; i++) {
        v[i] = (s(i) == 0);
    }
    return v;
}

vector<bool> Node::validMovesC4(State s) {
    vector<bool> v(maxMoves);
    for (int i = 0; i < maxMoves; i++) {
        v[i] = (s(i + 35) == 0);
    }
    return v;
}

vector<bool> Node::validMovesHex(State s) {
    vector<bool> v(maxMoves);
    for (int i = 0; i < maxMoves; i++) {
        v[i] = (s(i) == 0);
    }
    return v;
}

int choose(Vec distribution) {
    if (distribution.size() != maxMoves + 1) {
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
    cout << "Rare event occured when choosing random move\n";
    double bestProb = 0;
    int bestMove = -1;
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

int visited[125];
State dfsState(125); // for DFS purposes only
vector<int> adj[125];
State Node::startState; // definition for startState

void Node::initHex() { // this should be in Node.cpp
    Node::startState = Vec::Zero(122);
    Node::startState(121) = 1; // player trying to connect vertically
    // initialize adj
    // (x, y) -> 11y + x
    // (0,0) top left corner
    // 121 top edge, 122 left, 123 bottom, 124 right
    for (int i = 0; i < 121; i++) {
        int x = i % 11, y = i / 11;
        if (y == 0) { // top
            // adj[i].push_back(121);
            adj[121].push_back(i);
        } else {
            adj[i].push_back(i - 11);
        }
        if (x == 0) { // left
            // adj[i].push_back(122);
            adj[122].push_back(i);
        } else {
            adj[i].push_back(i - 1);
        }
        if (y == 10) { // bottom
            adj[i].push_back(123);
        } else {
            adj[i].push_back(i + 11);
        }
        if (x == 10) { // right
            adj[i].push_back(124);
        } else {
            adj[i].push_back(i + 1);
        }
        if (x != 0 && y != 10) { // diagonally down-left
            adj[i].push_back(i + 10);
        }
        if (x != 10 && y != 0) { // diagonally up-right
            adj[i].push_back(i - 10);
        }
    }
}

void dfs(int v, int player) {
    visited[v] = 1;
    for (const int &u : adj[v]) {
        if (visited[u] == 0 && dfsState[u] == player) {
            dfs(u, player);
        }
    }
}

pair<bool, double> Node::evaluateStateHex(State s) {
    dfsState << s, -s(121), s(121), -s(121);
    memset(visited, 0, sizeof(visited));
    dfs(121, s(121));
    if (visited[123]) {
        return make_pair(true, s[121]);
    }
    memset(visited, 0, sizeof(visited));
    dfs(122, -s(121));
    if (visited[124]) {
        return make_pair(true, -s[121]);
    }
    return make_pair(false, 0);
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

State Node::nextStateHex(State s, int move) {
    State copy = s;
    if (move < 0 || move >= 121 || copy(move) != 0) // optimize?
        cout << "Error: Invalid move\n";
    copy(move) = 1;
    return copy;
}

Node::Node(State s, Node *p) {
    if (s.rows() != stateSize) {
        cout << "Error: node initialized with invalid state\n";
    }
    state = s;
    parent = p;
    valid = validMovesHex(state);
    auto pair = evaluateStateHex(state);
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
Node *Node::chooseBest() {
    if (leaf) {
        cout << "Error: choosing from leaf node\n";
        return nullptr;
    }
    double totalVisits = 0;
    for (int i = 0; i < maxMoves; i++)
        totalVisits += N[i];
    double bestVal = -2;
    int bestMove = -1;
    for (int i = 0; i < maxMoves; i++) {
        if (valid[i]) {
            double val = Q[i] + c_puct * P[i] * sqrt(totalVisits + 1) / (N[i] + 1); // when totalVisits = 0, chooses first valid move?
            if (val > bestVal) {
                bestVal = val;
                bestMove = i;
            }
        }
    }
    if (bestMove == -1) {
        cout << "Error: No best move found\n"; // Probabilities might be nan!
        printf("State: ");
        for (int i = 0; i < stateSize; i++) {
            cout << state(i) << ", ";
        }
        cout << "\n";
        for (int i = 0; i < maxMoves; i++) {
            if (isnan(P[i])) {
                cout << "Probabilities are nan!\n";
                return nullptr;
            }
        }
        return nullptr;
    }
    return children[bestMove];
}

void Node::expand(Vec prob) { // expands node
    if (!leaf) {
        cout << "Error: tried to expand non-leaf node\n";
        return;
    }
    if (prob.rows() != maxMoves + 1) { // the last element is the predicted state value, not part of the probability distribution
        cout << "Error: probability vector size does not match\n";
        cout << "Probability size is " << prob.rows() << "\n";
        return;
    }
    leaf = false;
    for (int i = 0; i < maxMoves; i++) {
        if (valid[i]) {
            State copy = nextStateHex(state, i);
            copy = -copy; // switch to opponent's point of view, game-dependent
            children[i] = new Node(copy, this);
            P[i] = prob(i);
        } else {
            children[i] = nullptr;
        }
    }
}

void Node::update(double v, Node *child) {
    for (int i = 0; i < maxMoves; i++) {
        if (children[i] == child) {
            N[i]++;
            W[i] += v;
            Q[i] = W[i] / N[i];
        }
    }
}

Vec Node::getProbDistribution() {
    Vec v(maxMoves + 1); // last element is reserved for game winner, which is added later
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

Node *Node::chooseNewState() {
    if (leaf)
        cout << "Error: choosing move from leaf node\n";
    return children[chooseMove()];
}
