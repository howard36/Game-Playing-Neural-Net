#include "../Headers/Macros.h"
#include "../Headers/FullyConnectedLayer.h"
#include "../Headers/ActivationFunction.h"
#include "../Headers/Neural Net.h"
#include "../Headers/Node.h"

typedef FullyConnectedLayer<SigmoidActivationFunction> SigmoidLayer;
typedef FullyConnectedLayer<SoftMaxActivationFunction> SoftMaxLayer;
typedef FullyConnectedLayer<CustomActivationFunction> CustomLayer;

int main()
{
	srand(time(NULL));
	Node::initHex();
	/*
	vector<Layer*> layers;
	layers.push_back(new SigmoidLayer(stateSize, 200));
	layers.push_back(new SigmoidLayer(200, 200));
	layers.push_back(new CustomLayer(200, maxMoves+1));
	Network2 n("Hex 2 (100-100)", layers, 16, 0.05, 0.5);
	*/
	Network2 n2("Hex 2 (100-100)");

	printf("Start Training\n");
	n2.train(50, 50);

	cout << "Done!\n";
    return 0;
}
