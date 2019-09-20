#include "../Headers/ActivationFunction.h"
#include "../Headers/FullyConnectedLayer.h"
#include "../Headers/Macros.h"
#include "../Headers/Neural Net.h"
#include "../Headers/Node.h"

typedef FullyConnectedLayer<SigmoidActivationFunction> SigmoidLayer;
typedef FullyConnectedLayer<SoftMaxActivationFunction> SoftMaxLayer;
typedef FullyConnectedLayer<CustomActivationFunction> CustomLayer;

int main() {
    srand(time(NULL));
    Node::initC4(); // game dependent

    // vector<Layer *> layers;
    // layers.push_back(new SigmoidLayer(stateSize, 80));
    // layers.push_back(new SigmoidLayer(80, 50));
    // layers.push_back(new CustomLayer(50, maxMoves + 1));
    // Network2 n4("Hex 4 (80-50))", layers, 32, 0.01, 0.8);
    
    // Network2 n1("Hex 1 (100-100-100)");
    // Network2 n2("Hex 2 (100-100)");
    // Network2 n3("Hex 3 (60-60-60)");
    // Network2 n4("Hex 4 (80-50))");

    Network2 n("Connect4 3 (500-500) branch 5.2");
    
    n.play();

    // trbatch data;
    // fight(n2, n3, 50, 500, data);

    printf("Start Training\n");
    // n4.train(50, 50);

    cout << "Done!\n";
    return 0;
}
