#pragma once

#include "Macros.h"

class Node {
  private:
    Node *children[maxMoves];
    Node *parent;
    State state;
    vector<bool> valid;         // whether a move is valid
    int N[maxMoves];            // visit count
    double W[maxMoves];         // total action value
    double Q[maxMoves];         // mean action value
    double P[maxMoves];         // prior probability
    bool leaf;                  // whether this is currently a leaf node in the game tree
    bool end;                   // whether this is an ending state
    double endVal;              // the winner (if it is an end state)
    const double c_puct = 0.01; // change?

  public:
    Node(State s, Node *p);

    ~Node();

    // used for MCTS simulations
    Node *chooseBest();

    void expand(Vec prob);

    void update(double v, Node *child);

    Vec getProbDistribution();

    // used for game after simulations are complete
    // based on visit counts
    int chooseMove();

    Node *chooseNewState();

    static pair<bool, double> evaluateStateTTT(State s);
    static pair<bool, double> evaluateStateC4(State s);
    static pair<bool, double> evaluateStateHex(State s);

    static State nextStateTTT(State s, int move);
    static State nextStateC4(State s, int move);
    static State nextStateHex(State s, int move);

    static vector<bool> validMovesTTT(State s);
    static vector<bool> validMovesC4(State s);
    static vector<bool> validMovesHex(State s);

    static State startState; // this is only a declaration
    static void initC4();
    static void initHex();

    bool isLeaf() { return leaf; }
    Node *getParent() { return parent; }
    State getState() { return state; }
    bool isEndState() { return end; }
    double getEndVal() { return endVal; }
};
