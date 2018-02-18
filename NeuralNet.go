package main

import (
	"fmt"
	"math"
)

// weights: The weights w^{l}_ij are ordered by layer l, and for each layer,
// by i, then j in w^{l}_ij.
// Example: w^{1}_00, w^{1}_01, ..., w^{1}_0m, w^{1}_10, ..., w^{1}_1m, ..., w^{1}_n0, ..., w^{1}_nm,
//          w^{2}_00, w^{2}_01, ..., w^{2}_0m, w^{2}_10, ..., w^{2}_1m, ..., w^{2}_n0, ..., w^{2}_nm,
type Network struct {
	layers      []int
	activations []float64
	biases      []float64
	weights     []float64
}

func sum(xs []int) int {
	sum := 0
	for _, i := range xs {
		sum += i
	}
	return sum
}

func getNumberOfWeights(xs []int) int {
	n := 0
	x1 := xs[0]
	for i := 1; i < len(xs); i++ {
		x2 := xs[i]
		n += x1 * x2
		x1 = x2
	}
	return n
}

func CreateNetwork(layers []int) Network {
	nActivations := sum(layers)
	nBiases := sum(layers[1:])
	nWeights := getNumberOfWeights(layers)
	return Network{layers: layers, activations: make([]float64, nActivations), biases: make([]float64, nBiases), weights: make([]float64, nWeights)}
}

func Sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

func (n Network) getActivationBaseIndex(layer int) int {
	bi := 0
	for idx, n := range n.layers {
		if idx >= layer {
			break
		}
		bi += n
	}
	return bi
}

func (n Network) GetActivationIndex(index int, layer int) int {
	if layer >= len(n.layers) {
		panic(fmt.Sprintf("Activation layer index=%v must be smaller than the number of layers=%v", layer, len(n.layers)))
	}
	if index >= n.layers[layer] {
		panic(fmt.Sprintf("Activation index i=%v must be smaller than the number of activations=%v in layer %v", index, n.layers[layer], layer))
	}
	bi := n.getActivationBaseIndex(layer)
	return bi + index
}

func (n Network) GetActivation(index int, layer int) *float64 {
	return &n.activations[n.GetActivationIndex(index, layer)]
}

func (n Network) getBiasBaseIndex(layer int) int {
	bi := 0
	for idx, n := range n.layers {
		if idx == 0 {
			continue
		}
		if idx >= layer {
			break
		}
		bi += n
	}
	return bi
}

func (n Network) GetBiasIndex(index int, layer int) int {
	if layer >= len(n.layers) {
		panic(fmt.Sprintf("Bias layer index=%v must be smaller than the number of layers=%v", layer, len(n.layers)))
	}
	if index >= n.layers[layer] {
		panic(fmt.Sprintf("Bias index i=%v must be smaller than the number of activations=%v in layer %v", index, n.layers[layer], layer))
	}
	bi := n.getBiasBaseIndex(layer)
	return bi + index
}

func (n Network) GetBias(index int, layer int) float64 {
	return n.biases[n.GetBiasIndex(index, layer)]
}

// Start index of w^{l}_ij, i.e. linear index of w^{layer}_00 in
// n.weights
func (n Network) getWeightBaseIndex(layer int) int {
	return getNumberOfWeights(n.layers[0:layer])
}

func (n Network) GetWeightIndex(i int, j int, layer int) int {
	// Remember the meaning of the indices: w_ij^{l) is the weight from
	// neuron a_j^{l-1} to neuron a_i^{l}.
	if layer == 0 {
		panic(fmt.Sprintf("Weight layer index=%v must be bigger than 0 and smaller than the number of layers=%v", layer, len(n.layers)))
	}
	if layer >= len(n.layers) {
		panic(fmt.Sprintf("Weight layer index=%v must be smaller than the number of layers=%v", layer, len(n.layers)))
	}
	if i >= n.layers[layer] {
		panic(fmt.Sprintf("Weight index i=%v must be smaller than the number of activations=%v in layer %v", i, n.layers[layer], layer))
	}
	if j >= n.layers[layer-1] {
		panic(fmt.Sprintf("Weight index j=%v must be smaller than the number of activations=%v in layer %v", j, n.layers[layer-1], layer-1))
	}
	bi := n.getWeightBaseIndex(layer)
	nl1 := n.layers[layer-1]
	bi = bi + i*nl1
	return bi + j
}

func (n Network) GetWeight(i int, j int, layer int) float64 {
	return n.weights[n.GetWeightIndex(i, j, layer)]
}

func (n *Network) CalculateZ(i int, layer int) float64 {
	var z float64
	nPrevLayer := n.layers[layer-1]
	for j := 0; j < nPrevLayer; j++ {
		a_j := n.GetActivation(j, layer-1)
		w_ij := n.GetWeight(i, j, layer)
		z += w_ij * *a_j
	}
	b := n.GetBias(i, layer)
	z += b
	return z
}

func (n *Network) FeedforwardActivation(i int, layer int) float64 {
	if layer == 0 || layer >= len(n.layers) {
		panic(fmt.Sprintf("Activation layer index=%v must be bigger than 0 and smaller than the number of layers=%v", layer, len(n.layers)))
	}
	z := n.CalculateZ(i, layer)
	a := Sigmoid(z)
	return a
}

func (n *Network) FeedforwardLayer(layer int) {
	if layer == 0 {
		return
	}
	nLayer := n.layers[layer]
	for i := 0; i < nLayer; i++ {
		a := n.FeedforwardActivation(i, layer)
		a_i := n.GetActivation(i, layer)
		*a_i = a
	}
}

func (n *Network) Feedforward() {
	for layer := range n.layers {
		if layer == 0 {
			continue
		}
		n.FeedforwardLayer(layer)
	}
}

func (n *Network) CalculateOutputError() []float64 {

}

func (n *Network) SetInputLayer(inputActivations []float64) {
	// size check
}

func (n *Network) Backpropagate(nabla_L []float64) [][]float64 {
	// size check
}

type m struct {
	nabla [][]float64
}

func (network *Network) UpdateNetwork(nablas []m) {

}

func (n *Network) Solve(trainingSamples []TrainingSample) {
	// multiple epochs (i.e. outer loop does this multiple times and checks when NN is good enough)

	// randomize training samples (i.e. randomize arrays [1..size(training samples)]

	// chunk training samples into size m

	// for each x in m
	// 		- feed forward x
	// 		- calculate error for layer L
	//		- backprop, gives errors for all layers
	// update weights and biases for all x in m

	sizeMiniBatch := 20
	numMiniBatches := len(trainingSamples) / sizeMiniBatch

	ms := make([]m, sizeMiniBatch)

	//	ar opts = []struct
	lastLayerIdx := len(n.layers) - 1

	for j := 0; j < numMiniBatches; j++ {
		for i := 0; i < sizeMiniBatch; i++ {
			x := trainingSamples[j*sizeMiniBatch+i]
			n.SetInputLayer(x.inputActivations)
			n.Feedforward()
			ms[i].nabla[lastLayerIdx] = n.CalculateOutputError()
			ms[i].nabla = n.Backpropagate(ms[i].nabla[lastLayerIdx])
			// store
		}
		n.UpdateNetwork(ms)
	}

	// the rest
	//for i := numMiniBatches * sizeMiniBatch; i < len(trainingSamples); i++ {
	//	x := trainingSamples[i]
	//	n.SetInputLayer(x.inputActivations)
	//	n.Feedforward()
	//	nabla_L := n.CalculateOutputError()
	//	nabla := n.Backpropagate(nabla_L)
	//	// store
	//}
	//n.UpdateNetwork(nabla)

}
