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

func CreateNetwork(layers []int) Network {
	nBiases := sum(layers[1:])
	return Network{layers: layers, biases: make([]float64, nBiases), w: make([]float64, nWeights(layers))}
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
		panic(fmt.Sprintf("Layer index=%v must be smaller than the number of layers=%v", layer, len(n.layers)))
	}
	if index >= n.layers[layer] {
		panic(fmt.Sprintf("Activation index i=%v must be smaller than the number of activations=%v in layer %v", index, n.layers[layer], layer))
	}
	bi := n.getActivationBaseIndex(layer)
	return bi + index
}

func (n *Network) GetActivation(index int, layer int, mb *Minibatch) *float64 {
	aIdx := n.GetActivationIndex(index, layer)
	return &mb.a[aIdx]
}

func (n *Network) getNablaBaseIndex(layer int) int {
	return n.getActivationBaseIndex(layer)
}

func (n *Network) GetNablaIndex(index int, layer int) int {
	if l := len(n.layers); layer >= l {
		panic(fmt.Sprintf("Layer index=%v must be smaller than the number of layers=%v", layer, l))
	}
	if l := n.layers[layer]; index >= l {
		panic(fmt.Sprintf("Nabla index i=%v must be smaller than the number of activations=%v in layer %v", index, l, layer))
	}
	bi := n.getNablaBaseIndex(layer)
	return bi + index
}

func (n *Network) GetNabla(index int, layer int, mb *Minibatch) float64 {
	nablaIdx := n.GetNablaIndex(index, layer)
	return &mb.nabla[nablaIdx]
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

func (n *Network) GetWeight(i int, j int, layer int, mb *Minibatch) float64 {
	wIdx := n.GetWeightIndex(i, j, layer)
	return mb.w[wIdx]
}

func (n *Network) CalculateZ(i int, layer int, mb *Minibatch) float64 {
	var z float64
	nPrevLayer := n.layers[layer-1]
	for j := 0; j < nPrevLayer; j++ {
		a_j := n.GetActivation(j, layer-1, mb)
		w_ij := n.GetWeight(i, j, layer, mb)
		z += w_ij * *a_j
	}
	b := n.GetBias(i, layer)
	z += b
	return z
}

func (n *Network) FeedforwardActivation(i int, layer int, mb *Minibatch) float64 {
	if l := len(n.layers); layer == 0 || layer >= l {
		panic(fmt.Sprintf("Activation layer index=%v must be bigger than 0 and smaller than the number of layers=%v", layer, l))
	}
	z := n.CalculateZ(i, layer, mb)
	a := Sigmoid(z)
	return a
}

func (n *Network) FeedforwardLayer(layer int, mb *Minibatch) {
	if layer == 0 {
		return
	}
	nLayer := n.layers[layer]
	for i := 0; i < nLayer; i++ {
		a := n.FeedforwardActivation(i, layer, mb)
		a_i := n.GetActivation(i, layer, mb)
		*a_i = a
	}
}

func (n *Network) Feedforward(mb *Minibatch) {
	for layer := range n.layers {
		if layer == 0 {
			continue
		}
		n.FeedforwardLayer(layer, mb)
	}
}

func (n *Network) CalculateErrorInOutputLayer(expectedOutputActivations []float64, mb *Minibatch) {
	// Equation (BP1) and (30), Chapter 2 of http://neuralnetworksanddeeplearning.com
	outputLayerIdx := len(n.layers) - 1
	if len(expectedOutputActivations) != n.layers[outputLayerIdx] {
		panic(fmt.Sprintf("Expected output activation size %v does not match number of activations %v in ouput layer", len(expectedOutputActivations), n.layers[outputLayerIdx]))
	}
	nActivations := n.layers[outputLayerIdx]
	output := mb.nabla[n.getActivationBaseIndex(outputLayerIdx):]
	for i := 0; i < nActivations; i++ {
		a_i := n.GetActivation(i, outputLayerIdx, mb)
		z_i := n.CalculateZ(i, outputLayerIdx, mb)
		output[i] = (*a_i - expectedOutputActivations[i]) * SigmoidPrime(z_i)
	}
}

func (n *Network) SetInputActivations(inputActivations []float64, mb *Minibatch) {
	if len(inputActivations) != n.layers[0] {
		panic(fmt.Sprintf("Input activation size %v does not match number of activations %v in input layer", len(inputActivations), n.layers[0]))
	}
	for idx, a := range inputActivations {
		a_i := n.GetActivation(idx, 0, mb)
		*a_i = a
	}
}

func (n *Network) Backpropagate(mb *Minibatch) {
	// Equation (45), Chapter 2 of http://neuralnetworksanddeeplearning.com
	outputLayerIdx := len(n.layers) - 1
	for layer := outputLayerIdx - 1; layer > 0; layer-- {
		nActivations := n.layers[layer]
		nNextActivations := n.layers[layer+1]
		abi1 := n.getActivationBaseIndex(layer)
		nabla := mb.nabla[abi1:abi1+nActivations]
		abi2 := n.getActivationBaseIndex(layer+1)
		nablaNext := mb.nabla[abi2:abi2+nNextActivations]
		for j := 0; j < nActivations; j++ {
			var tmp float64
			for k := 0; k < nNextActivations; k++ {
				weight_kj := n.GetWeight(k, j, layer+1, mb)
				tmp += weight_kj * nablaNext[k]
			}
			z_j := n.CalculateZ(j, layer, mb)
			s := SigmoidPrime(z_j)
			nabla[j] = tmp * s
		}
	}
}

func (n *Network) UpdateNetwork(mbs []Minibatch) {
	eta := 0.1
	nMiniBatches := len(mbs)
	for layer := range n.layers {
		if layer == 0 {
			continue
		}

		for j := 0; j < n.layers[layer]; j++ {
			// w_j^l
			w_jk := n.GetWeight()
			dw := make([]float64, n.layers[layer-1])
			var dw float64
			for mbIdx := 0; mbIdx < nMiniBatches; mbIdx++ {
				for aIdx := 0; aIdx < n.layers[layer]; aIdx++ {
					a := n.GetActivation(aIdx, layer, &mbs[mbIdx])
					nabla := n.GetNabla(aIdx, layer, &mbs[mbIdx])
					dw += *a * nabla
				}
			}
			dw := eta / float64(nMiniBatches) * dw
			w_jk -= dw
			n.weights[n.GetWeightIndex(j, k, layer)] = w_jk











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
	nMiniBatches := len(trainingSamples) / sizeMiniBatch
	mbs := make([]Minibatch, nMiniBatches)
	for j := 0; j < nMiniBatches; j++ {
		for i := 0; i < sizeMiniBatch; i++ {
			mb := mbs[j]
			x := trainingSamples[j*sizeMiniBatch+i]

			// TODO SS: also set weights from reference!


			n.SetInputActivations(x.inputActivations, &mb)
			n.Feedforward(&mb)
			n.CalculateErrorInOutputLayer(x.outputActivations, &mb)
			n.Backpropagate(&mb)
		}
		n.UpdateNetwork(mbs)
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
