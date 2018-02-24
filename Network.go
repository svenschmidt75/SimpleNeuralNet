package main

import (
	"fmt"
	"math"
)
// weights: The weights w_ij^{l} are ordered by layer l, and for each layer,
// by i, then j.
// Example: w_00^{1}, w_01^{1}, ..., w_0m^{1}, w_10^{1}, ..., w_1m^{1}, ..., w_n0^{1}, ..., w_nm^{1},
//          w_00^{2}, w_01^{2}, ..., w_0m^{2}, w_10^{2}, ..., w_1m^{2}, ..., w_n0^{2}, ..., w_nm^{2},
type Network struct {
	// how many activations per layer
	layers  []int

	// biases, ordered by layer, then index of activation
	biases  []float64

	// the unmodified weights that the minibatches are initialized with
	w []float64

	// minibatches for Stochastic Gradient Decent
	batches []Minibatch
}

func sum(xs []int) int {
	sum := 0
	for _, i := range xs {
		sum += i
	}
	return sum
}

func nWeights(xs []int) int {
	n := 0
	x1 := xs[0]
	for i := 1; i < len(xs); i++ {
		x2 := xs[i]
		n += x1 * x2
		x1 = x2
	}
	return n
}

func nActivations(xs []int) int {
	return sum(xs)
}

func CreateNetwork(layers []int, nMiniBatches int) Network {
	batches := make([]Minibatch, nMiniBatches)
	for mbIdx, _ := range batches {
		mb := &batches[mbIdx]
		mb.a = make([]float64, nActivations(layers))
		mb.z = make([]float64, nActivations(layers))
		mb.w = make([]float64, nWeights(layers))
	}
	nBiases := sum(layers[1:])
	return Network{layers: layers, biases: make([]float64, nBiases), w: make([]float64, nWeights(layers)), batches: batches}
}

func Sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

func SigmoidPrime(z float64) float64 {
	// derivative of the sigmoid function
	return Sigmoid(z) * (1 - Sigmoid(z))
}

func (n *Network) getActivationBaseIndex(layer int) int {
	bi := sum(n.layers[0:layer])
	return bi
}

func (n *Network) GetActivationIndex(index int, layer int) int {
	if layer >= len(n.layers) {
		panic(fmt.Sprintf("Activation layer index=%v must be smaller than the number of layers=%v", layer, len(n.layers)))
	}
	if index >= n.layers[layer] {
		panic(fmt.Sprintf("Activation index i=%v must be smaller than the number of activations=%v in layer %v", index, n.layers[layer], layer))
	}
	bi := n.getActivationBaseIndex(layer)
	return bi + index
}

func (n *Network) GetActivation(index int, layer int, miniBatchIndex int) *float64 {
	aIdx := n.GetActivationIndex(index, layer)
	return &n.batches[miniBatchIndex].a[aIdx]
}

func (n *Network) getBiasBaseIndex(layer int) int {
	bi := sum(n.layers[1:layer])
	return bi
}

func (n *Network) GetBiasIndex(index int, layer int) int {
	if layer >= len(n.layers) {
		panic(fmt.Sprintf("Bias layer index=%v must be smaller than the number of layers=%v", layer, len(n.layers)))
	}
	if index >= n.layers[layer] {
		panic(fmt.Sprintf("Bias index i=%v must be smaller than the number of activations=%v in layer %v", index, n.layers[layer], layer))
	}
	bi := n.getBiasBaseIndex(layer)
	return bi + index
}

func (n *Network) GetBias(index int, layer int) float64 {
	return n.biases[n.GetBiasIndex(index, layer)]
}

// Start index of w^{l}_ij, i.e. linear index of w^{layer}_00 in
// n.weights
func (n *Network) getWeightBaseIndex(layer int) int {
	return nWeights(n.layers[0:layer])
}

func (n *Network) GetWeightIndex(i int, j int, layer int) int {
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

func (n *Network) GetRefWeight(i int, j int, layer int) float64 {
	return n.w[n.GetWeightIndex(i, j, layer)]
}

func (n *Network) GetWeight(i int, j int, layer int, miniBatchIndex int) float64 {
	wIdx := n.GetWeightIndex(i, j, layer)
	return n.batches[miniBatchIndex].w[wIdx]
}

func (n *Network) CalculateZ(i int, layer int, miniBatchIndex int) float64 {
	var z float64
	nPrevLayer := n.layers[layer-1]
	for j := 0; j < nPrevLayer; j++ {
		a_j := n.GetActivation(j, layer-1, miniBatchIndex)
		w_ij := n.GetWeight(i, j, layer, miniBatchIndex)
		z += w_ij * *a_j
	}
	b := n.GetBias(i, layer)
	z += b
	return z
}

func (n *Network) FeedforwardActivation(i int, layer int, miniBatchIndex int) float64 {
	if l := len(n.layers); layer == 0 || layer >= l {
		panic(fmt.Sprintf("Activation layer index=%v must be bigger than 0 and smaller than the number of layers=%v", layer, l))
	}
	if l := len(n.batches); miniBatchIndex >= l {
		panic(fmt.Sprintf("Minibatch index=%v must be between 0 and =%v", miniBatchIndex, l))
	}
	z := n.CalculateZ(i, layer, miniBatchIndex)
	a := Sigmoid(z)
	return a
}

func (n *Network) FeedforwardLayer(layer int, miniBatchIndex int) {
	if layer == 0 {
		return
	}
	nLayer := n.layers[layer]
	for i := 0; i < nLayer; i++ {
		a := n.FeedforwardActivation(i, layer, miniBatchIndex)
		a_i := n.GetActivation(i, layer, miniBatchIndex)
		*a_i = a
	}
}

func (n *Network) Feedforward(miniBatchIndex int) {
	for layer := range n.layers {
		if layer == 0 {
			continue
		}
		n.FeedforwardLayer(layer, miniBatchIndex)
	}
}

func (n *Network) CalculateErrorInOutputLayer(expectedOutputActivations []float64, miniBatchIndex int) []float64 {
	// Equation (BP1) and (30), Chapter 2 of http://neuralnetworksanddeeplearning.com
	outputLayerIdx := len(n.layers) - 1
	if len(expectedOutputActivations) != n.layers[outputLayerIdx] {
		panic(fmt.Sprintf("Expected output activation size %v does not match number of activations %v in ouput layer", len(expectedOutputActivations), n.layers[outputLayerIdx]))
	}
	nActivations := n.layers[outputLayerIdx]
	output := make([]float64, nActivations)
	for i := 0; i < nActivations; i++ {
		a_i := n.GetActivation(i, outputLayerIdx, miniBatchIndex)
		z_i := n.CalculateZ(i, outputLayerIdx, miniBatchIndex)
		output[i] = (*a_i - expectedOutputActivations[i]) * SigmoidPrime(z_i)
	}
	return output
}

func (n *Network) SetInputActivations(inputActivations []float64, miniBatchIndex int) {
	if len(inputActivations) != n.layers[0] {
		panic(fmt.Sprintf("Input activation size %v does not match number of activations %v in input layer", len(inputActivations), n.layers[0]))
	}
	for idx, a := range inputActivations {
		a_i := n.GetActivation(idx, 0, miniBatchIndex)
		*a_i = a
	}
}

func (n *Network) Backpropagate(nabla_L []float64) [][]float64 {
	// Equation (45), Chapter 2 of http://neuralnetworksanddeeplearning.com
	outputLayerIdx := len(n.layers) - 1
	if len(nabla_L) != n.layers[outputLayerIdx] {
		panic(fmt.Sprintf("Output error size %v does not match number of activations %v in output layer", len(nabla_L), n.layers[outputLayerIdx]))
	}
	nablas := make([][]float64, len(n.layers)-1)
	nablas[outputLayerIdx-1] = nabla_L
	for layer := outputLayerIdx - 1; layer > 0; layer-- {
		nablas[layer-1] = make([]float64, n.layers[layer])
		nActivations := n.layers[layer]
		nNextActivations := n.layers[layer+1]
		for j := 0; j < nActivations; j++ {
			var tmp float64
			for k := 0; k < nNextActivations; k++ {
				weight_kj := n.GetWeight(k, j, layer+1)
				nabla_k := nablas[layer][k]
				tmp += weight_kj * nabla_k
			}
			z_j := n.CalculateZ(j, layer)
			s := SigmoidPrime(z_j)
			nablas[layer-1][j] = tmp * s
		}
	}
	return nablas
}

type m struct {
	nabla [][]float64
}

func (n *Network) UpdateNetwork(nablas []m) {
	eta := 0.1
	miniBatchSize := len(nablas)
	for layer := range n.layers {
		if layer == 0 {
			continue
		}

		// sum over activations a_j^l
		for j := 0; j < n.layers[layer]; j++ {
			// bias
			dw := make([]float64, n.layers[layer-1])
			var db float64
			for batchIdx := range nablas {
				nabla_j := nablas[batchIdx].nabla[layer-1][j]
				for k := 0; k < n.layers[layer-1]; k++ {
					a_k := n.GetActivation(k, layer-1)
					dw[k] += *a_k * nabla_j
				}
				db += nabla_j
			}

			// weight
			for k, v := range dw {
				w_jk := n.GetWeight(j, k, layer)
				dw := eta / float64(miniBatchSize) * v
				w_jk -= dw
				n.weights[n.GetWeightIndex(j, k, layer)] = w_jk
			}

			b_j := n.GetBias(j, layer)
			n.biases[n.GetBiasIndex(j, layer)] = b_j - eta/float64(miniBatchSize)*db

		}
	}
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

	for j := 0; j < numMiniBatches; j++ {
		for i := 0; i < sizeMiniBatch; i++ {
			x := trainingSamples[j*sizeMiniBatch+i]
			n.SetInputActivations(x.inputActivations)
			n.Feedforward()
			nabla_L := n.CalculateErrorInOutputLayer(x.outputActivations)
			ms[i].nabla = n.Backpropagate(nabla_L)
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
