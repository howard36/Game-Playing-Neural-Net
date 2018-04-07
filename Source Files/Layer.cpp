#include "../Headers/Macros.h"
#include "../Headers/Layer.h"
#include "../Headers/FullyConnectedLayer.h"

Layer::Layer() {}

Layer::~Layer(){
//	printf("destroying layer\n");
}

Layer* read(ifstream& fin) {
	int layerType;
	fin >> layerType;
	if (layerType == 1) { // Fully connected layer
		return read_FC(fin);
	}
	else {
		cout << "Error: Invalid layer type when reading layer from file\n";
		return nullptr;
	}
}