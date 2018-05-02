#pragma once

#include "Macros.h"
#include "Layer.h"

class Node;

class Network2
{

public:
	Network2();
	//	Network2(const std::vector<Layer>& sizes, const checker_type& f, int batchSize, double _learnRate, double maxRate, double minRate, double L2, double momentum);
	Network2(const string _name);

	Network2(const string _name, const std::vector<Layer*>& _layers, const checker_type& ch, int _in, int _out, int mbs, double lr);

	Network2(Network2& n); // copy constructor

	~Network2();

//	void train(trbatch& data, trbatch& test, int numEpochs);
	void train(int sims, int games);

	void play();

	void feedForward(Mat& input) const; // pass by reference. input layer will output as output layer

	friend ifstream& operator>> (ifstream& fin, Network2& n);

	friend ofstream& operator<< (ofstream& f, const Network2& n);

	friend const Network2& fight(const Network2& n1, const Network2& n2, int sims, int games, trbatch& data);

	Network2& operator= (const Network2& n);

	void setChecker(const checker_type& ch);

private:

	// functions
	void selfPlay(trbatch& trainingData, int sims); // plays a game against itself to generate training data ofMCTS probability distributions

	void simulate(Node * const start) const; // go down the MCTS tree, based on moves chosen by the neural net

	pair<int, double> selectMove(const State& s, int moves, trbatch& data) const;
	
	void learn(trbatch& data);

	// properties
	checker_type checker;

	// the layers in the network
	std::vector<Layer*> layers;

	int numLayers, in, out;

	int age; // how many iterations it has been trained for

	int miniBatchSize;

	string name;

	// how quickly it learns
	double learnRate;
	
//	double maxRate, minRate;

	// how much L2regularization affects cost
	// if high, it will focus on keeping weights low
	// if low, it will focus on minimizing regular cost function
//	double L2;

//	double momentum;

	// to track progress
//	double maxfrac = 0;

	// random device class instance, source of 'true' randomness for initializing random seed
	std::random_device randDev;

	// Mersenne twister PRNG, initialized with seed from previous random device instance
	std::mt19937 randGen;
};