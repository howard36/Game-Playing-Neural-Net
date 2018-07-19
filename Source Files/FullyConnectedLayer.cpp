#include "../Headers/FullyConnectedLayer.h"
#include "../Headers/Macros.h"

#define FC_LAYER_TEMPLATE template<typename ActivationFn>

using namespace std;

FC_LAYER_TEMPLATE
FullyConnectedLayer<ActivationFn>::FullyConnectedLayer() {}

// currently, this is just a fully connected layer using sigmoid activation function

FC_LAYER_TEMPLATE
FullyConnectedLayer<ActivationFn>::FullyConnectedLayer(int _in, int _out) {
	in = _in;
	out = _out;

	// random
	randGen = mt19937(chrono::high_resolution_clock::now().time_since_epoch().count());
	// defaults to mean of 0.0, standard dev of 1.0
	randDistribution = normal_distribution<double>();

	weights.resize(out, in);
	biases.resize(out);

	for (int i = 0; i < out; ++i) {
		// set random weights
		for (int j = 0; j < in; ++j)
			weights(i, j) = randDistribution(randGen) / sqrt(in);
		// set random biases
		biases(i) = randDistribution(randGen);
	}
}

FC_LAYER_TEMPLATE
FullyConnectedLayer<ActivationFn>::FullyConnectedLayer(int _in, int _out, const Mat& _weights, const Vec& _biases) {
	in = _in;
	out = _out;
	weights = _weights;
	biases = _biases;
	if (weights.rows() != out || weights.cols() != in || biases.rows() != out) {
		cout << "Error: Invalid dimensions for weights and biases in FC Layer constructor\n";
	}
}

// destructor
FC_LAYER_TEMPLATE
FullyConnectedLayer<ActivationFn>::~FullyConnectedLayer(){
//	printf("FullyConnectedLayer destructor called\n");
}

FC_LAYER_TEMPLATE
void FullyConnectedLayer<ActivationFn>::apply(Mat& input) {
	prevActivations = input; // this is a^(l-1) in the tutorial
	pre = weights*input + biases.replicate(1, input.cols()); // these are the z-values in the tutorial
	activations = ActivationFn::activation(pre);
	derivs = ActivationFn::activationDeriv(pre);
	input = activations; // changes input directly, since it is passed by reference
}

// WTD is W^T x D, where W^T is the transpose of weight matrix, D is delta vector
FC_LAYER_TEMPLATE
void FullyConnectedLayer<ActivationFn>::computeDeltaLast(const Mat& output, const Mat& ans, Mat& WTD) {
//	delta = costDeriv(output, ans).cwiseProduct(derivs); // delta^L = grad_a(C) * sigma'(z^L)	(BP1)
	delta = output - ans; // hardcoded for softmax layer and cross entropy cost function! (BP1 assumes a_j only depends on z_j)
	delta.row(delta.rows()-1) = delta.row(delta.rows() - 1).cwiseProduct(derivs.row(derivs.rows()-1)); // for custom cost function
	WTD = weights.transpose() * delta; // this is needed to compute delta^(L-1)
}

FC_LAYER_TEMPLATE
void FullyConnectedLayer<ActivationFn>::computeDeltaBack(Mat& WTD) {
	delta = WTD.cwiseProduct(derivs); // delta^l = ((W^(l+1))^T x delta^l) * sigma'(z)		(BP2)
	WTD = weights.transpose() * delta; // this is needed to compute delta^(l-1)
}

FC_LAYER_TEMPLATE
void FullyConnectedLayer<ActivationFn>::updateBiasAndWeights(double lrate) {
	biases -= lrate*delta.rowwise().mean(); // (BP3)
	weights -= (lrate / delta.cols())*(delta * prevActivations.transpose()); // (BP4)
}

FC_LAYER_TEMPLATE
inline Mat FullyConnectedLayer<ActivationFn>::costDeriv(const Mat& output, const Mat& ans) {
	 return output - ans;
}

// for debugging purposes
FC_LAYER_TEMPLATE
void FullyConnectedLayer<ActivationFn>::print()
{
	cout << "weights:\n" << weights;
//	cout << "\ndelta:\n" << delta;
//	cout << "\nprevactivation:\n" << prevActivations;
	cout << "\npre\n" << pre;
//	cout << "\nderivs\n" << derivs;
//	cout << "\nBiases:\n" << biases;
}

FC_LAYER_TEMPLATE
inline pair<int, int> FullyConnectedLayer<ActivationFn>::getSize() { return make_pair(in, out); }

FC_LAYER_TEMPLATE
Layer* FullyConnectedLayer<ActivationFn>::copy() {
	if (ActivationFn::id() == 1) {
		return new FullyConnectedLayer<SigmoidActivationFunction>(in, out, weights, biases);
	}
	else if (ActivationFn::id() == 2) {
		return new FullyConnectedLayer<TanhActivationFunction>(in, out, weights, biases);
	}
	else if (ActivationFn::id() == 3) {
		return new FullyConnectedLayer<SoftMaxActivationFunction>(in, out, weights, biases);
	}
	else if (ActivationFn::id() == 4) {
		return new FullyConnectedLayer<CustomActivationFunction>(in, out, weights, biases);
	}
	else {
		cout << "Error: invalid activation id when copying layer\n";
		return nullptr;
	}
}

FC_LAYER_TEMPLATE
void FullyConnectedLayer<ActivationFn>::write(ofstream& fout) {
	fout << "1\n"; // layer type of a fully connected layer
	fout << ActivationFn::id() << "\n";
	fout << in << " " << out << "\n";
	fout << weights << "\n" << biases << "\n";
}

Layer* read_FC(ifstream& fin) {
	int activationType, in, out;
	fin >> activationType >> in >> out;
	if (activationType == 1) {
		auto layer = new FullyConnectedLayer<SigmoidActivationFunction>(in, out);
		for (int r = 0; r < out; r++) {
			for (int c = 0; c < in; c++) {
				fin >> (*layer).weights(r, c);
			}
		}
		for (int r = 0; r < out; r++) {
			fin >> (*layer).biases(r);
		}
		return layer;
	}
	else if (activationType == 2) {
		auto layer = new FullyConnectedLayer<TanhActivationFunction>(in, out);
		for (int r = 0; r < out; r++) {
			for (int c = 0; c < in; c++) {
				fin >> (*layer).weights(r, c);
			}
		}
		for (int r = 0; r < out; r++) {
			fin >> (*layer).biases(r);
		}
		return layer;
	}
	else if (activationType == 3) {
		auto layer = new FullyConnectedLayer<SoftMaxActivationFunction>(in, out);
		for (int r = 0; r < out; r++) {
			for (int c = 0; c < in; c++) {
				fin >> (*layer).weights(r, c);
			}
		}
		for (int r = 0; r < out; r++) {
			fin >> (*layer).biases(r);
		}
		return layer;
	}
	else if (activationType == 4) {
		auto layer = new FullyConnectedLayer<CustomActivationFunction>(in, out);
		for (int r = 0; r < out; r++) {
			for (int c = 0; c < in; c++) {
				fin >> (*layer).weights(r, c);
			}
		}
		for (int r = 0; r < out; r++) {
			fin >> (*layer).biases(r);
		}
		return layer;
	}
	else {
		cout << "Error: Invalid activation type when reading layer from file\n";
		return nullptr;
	}
}


template class FullyConnectedLayer<SigmoidActivationFunction>;
template class FullyConnectedLayer<TanhActivationFunction>;
template class FullyConnectedLayer<SoftMaxActivationFunction>;
template class FullyConnectedLayer<CustomActivationFunction>;
