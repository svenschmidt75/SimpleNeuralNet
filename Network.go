package main

import (
	"SimpleNeuralNet/MNISTImport"
	"fmt"
	"math"
	"math/rand"
	"time"
)

/* TODO:
 * - function that returns output layer index
 */

// weights: The weights w_ij^{l} are ordered by layer l, and for each layer,
// by i, then j.
// Example: w_00^{1}, w_01^{1}, ..., w_0m^{1}, w_10^{1}, ..., w_1m^{1}, ..., w_n0^{1}, ..., w_nm^{1},
//          w_00^{2}, w_01^{2}, ..., w_0m^{2}, w_10^{2}, ..., w_1m^{2}, ..., w_n0^{2}, ..., w_nm^{2},
type Network struct {
	// how many activations per layer
	layers []int

	// biases, ordered by layer, then index of activation
	biases []float64

	// weights
	weights []float64
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

func (n Network) nWeightsInLayer(layer int) int {
	if layer == 0 {
		panic("Layer 0 has no weights")
	}
	idx1 := nWeights(n.layers[0:layer])
	idx2 := nWeights(n.layers[0 : layer+1])
	return idx2 - idx1
}

func CreateNetwork(layers []int) Network {
	nBiases := sum(layers[1:])
	return Network{layers: layers, biases: make([]float64, nBiases), weights: make([]float64, nWeights(layers))}
}

func Sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

func SigmoidPrime(z float64) float64 {
	// derivative of the sigmoid function
	s := Sigmoid(z)
	return s * (1.0 - s)
}

func (n *Network) nWeights() int {
	return nWeights(n.layers)
}

func (n *Network) nActivations() int {
	return sum(n.layers)
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
	return n.GetActivationIndex(index, layer)
}

func (n *Network) GetNabla(index int, layer int, mb *Minibatch) float64 {
	nablaIdx := n.GetNablaIndex(index, layer)
	return mb.nabla[nablaIdx]
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

func (n *Network) GetWeight(i int, j int, layer int) float64 {
	return n.weights[n.GetWeightIndex(i, j, layer)]
}

func (n *Network) CalculateZ(i int, layer int, mb *Minibatch) float64 {
	var z float64
	nPrevLayer := n.layers[layer-1]
	for j := 0; j < nPrevLayer; j++ {
		a_j := n.GetActivation(j, layer-1, mb)
		w_ij := n.GetWeight(i, j, layer)
		z += w_ij * *a_j
		//fmt.Printf("%v, %v, %v\n", *a_j, w_ij, z)
	}
	//	fmt.Printf("z: %v\n", z)
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

func (n *Network) InitializeNetworkWeightsAndBiasesLayer(layer int) {
	if layer == 0 {
		return
	}
	rand.Seed(time.Now().Unix())
	// number of weights in this layer
	weightBaseIdx := n.getWeightBaseIndex(layer)
	nWeights := n.nWeightsInLayer(layer)
	for widx := 0; widx < nWeights; widx++ {
		n.weights[weightBaseIdx+widx] = rand.Float64() / 100.0
	}
	biasBaseIdx := n.getBiasBaseIndex(layer)
	nBiases := n.layers[layer]
	for bidx := 0; bidx < nBiases; bidx++ {
		n.biases[biasBaseIdx+bidx] = rand.Float64() / 100.0
	}
}

func (n *Network) InitializeNetworkWeightsAndBiases() {
	for layer := range n.layers {
		if layer == 0 {
			continue
		}
		n.InitializeNetworkWeightsAndBiasesLayer(layer)
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
		da := *a_i - expectedOutputActivations[i]
		z_i := n.CalculateZ(i, outputLayerIdx, mb)
		ds := SigmoidPrime(z_i)
		output[i] = da * ds
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

func (n *Network) BackpropagateError(mb *Minibatch) {
	// Equation (45), Chapter 2 of http://neuralnetworksanddeeplearning.com
	outputLayerIdx := len(n.layers) - 1
	for layer := outputLayerIdx - 1; layer > 0; layer-- {
		nActivations := n.layers[layer]
		nNextActivations := n.layers[layer+1]
		abi1 := n.getActivationBaseIndex(layer)
		nabla := mb.nabla[abi1 : abi1+nActivations]
		abi2 := n.getActivationBaseIndex(layer + 1)
		nablaNext := mb.nabla[abi2 : abi2+nNextActivations]
		for j := 0; j < nActivations; j++ {
			var tmp float64
			for k := 0; k < nNextActivations; k++ {
				weight_kj := n.GetWeight(k, j, layer+1)
				tmp += weight_kj * nablaNext[k]
			}
			z_j := n.CalculateZ(j, layer, mb)
			s := SigmoidPrime(z_j)
			error := tmp * s
			nabla[j] = error
			//			fmt.Printf("%v, %v, %v\n", z_j, s, error)
		}
	}
}

func (n *Network) CalculateDerivatives(mbs []Minibatch) ([]float64, []float64) {
	/* We use Stochastic Gradient Descent to update the weights and biases.
	 * The idea is that we sample m trainings samples from the entire set
	 * and feed forward, then backpropagate. We then update the weights and
	 * biases based on those m samples instead of updating after all training
	 * samples. The assumption is that those m samples are representative of
	 * all the training samples, which then saves time.
	 *
	 * So, we have
	 *
	 * w_{jk}^{l} -> w_{jk}^{l} - \eta \frac{\partial C}{\partial w_{jk}^{l}}
	 * and
	 * b_{j}^{l} -> b_{j}^{l} - \eta \frac{\partial C}{\partial b_{j}^{l}}
	 *
	 * Since we use Stochastic Gradient Decent, we approximate
	 * w_{jk}^{l} -> w_{jk}^{l} - \eta \frac{\partial C_{x}}{\partial w_{jk}^{l}}
	 * and
	 * b_{j}^{l} -> b_{j}^{l} - \eta \frac{\partial C_{x}}{\partial b_{j}^{l}}
	 * where
	 * \frac{\partial C}{\partial w_{jk}^{l}} \approx \frac{1}{m} \sum_{x} \frac{\partial C_{x}}{\partial b_{j}^{l}}
	 * and \frac{\partial C_{x}}{\partial b_{j}^{l}} = a_{k}^{l-1, x} \delta_{j}^{l, x}
	 */

	// d C_x / d_wjk^l
	dw := make([]float64, n.nWeights())

	// d C_x / d_bj^l
	db := make([]float64, n.nActivations())

	nMiniBatches := len(mbs)
	for layer := range n.layers {
		if layer == 0 {
			continue
		}
		nActivations := n.layers[layer]
		nPrevActivations := n.layers[layer-1]
		for j := 0; j < nActivations; j++ {
			for k := 0; k < nPrevActivations; k++ {
				// w_jk^l
				var dw_jk float64
				for mbIdx := range mbs {
					mb := mbs[mbIdx]
					a := n.GetActivation(k, layer-1, &mb)
					nabla := n.GetNabla(j, layer, &mb)
					dCx_dw := *a * nabla
					dw_jk += dCx_dw
				}
				dw_jk /= float64(nMiniBatches)
				wIdx := n.GetWeightIndex(j, k, layer)
				dw[wIdx] = dw_jk
			}
			// b_j^l
			var db_j float64
			for mbIdx := range mbs {
				mb := mbs[mbIdx]
				nabla := n.GetNabla(j, layer, &mb)
				db_j += nabla
			}
			db_j /= float64(nMiniBatches)
			bIdx := n.GetBiasIndex(j, layer)
			db[bIdx] = db_j
		}
	}
	return dw, db
}

func (n *Network) UpdateNetwork(eta float64, dw []float64, db []float64) {
	for layer := range n.layers {
		if layer == 0 {
			continue
		}
		nActivations := n.layers[layer]
		nPrevActivations := n.layers[layer-1]
		for j := 0; j < nActivations; j++ {
			for k := 0; k < nPrevActivations; k++ {
				// w_jk^l
				wIdx := n.GetWeightIndex(j, k, layer)
				dw_jk := dw[wIdx]
				w_jk := n.GetWeight(j, k, layer)
				w_jk -= eta * dw_jk
				n.weights[wIdx] = w_jk
			}
			// b_j^l
			bIdx := n.GetBiasIndex(j, layer)
			db_j := db[bIdx]
			b_j := n.GetBias(j, layer)
			b_j -= eta * db_j
			n.biases[bIdx] = b_j
		}
	}
}

func generateRandomIndices(size int) []int {
	rand.Seed(time.Now().UTC().UnixNano())
	// generate random permutation
	perm := rand.Perm(size)
	return perm
}

func max(lhs int, rhs int) int {
	if lhs < rhs {
		return rhs
	}
	return lhs
}

func min(lhs int, rhs int) int {
	if lhs < rhs {
		return lhs
	}
	return rhs
}

func (n *Network) Train(trainingSamples []MNISTImport.TrainingSample, epochs int, eta float64) {
	// Stochastic Gradient Decent
	sizeMiniBatch := min(len(trainingSamples), 20)
	nMiniBatches := len(trainingSamples) / sizeMiniBatch
	mbs := CreateMiniBatches(sizeMiniBatch, n.nActivations(), n.nWeights())

	for epoch := 0; epoch < epochs; epoch++ {
		indices := generateRandomIndices(len(trainingSamples))
		for j := 0; j < nMiniBatches; j++ {
			for i := 0; i < sizeMiniBatch; i++ {
				mb := mbs[i]
				index := indices[j*sizeMiniBatch+i]
				x := trainingSamples[index]
				n.SetInputActivations(x.InputActivations, &mb)
				n.Feedforward(&mb)
				n.CalculateErrorInOutputLayer(x.OutputActivations, &mb)
				n.BackpropagateError(&mb)
			}
			dw, db := n.CalculateDerivatives(mbs)
			n.UpdateNetwork(eta, dw, db)
		}
		// the rest
		nRest := len(trainingSamples) - sizeMiniBatch*nMiniBatches
		if nRest > 0 {
			for i := 0; i < nRest; i++ {
				mb := mbs[i]
				index := indices[nMiniBatches*sizeMiniBatch+i]
				x := trainingSamples[index]
				n.SetInputActivations(x.InputActivations, &mb)
				n.Feedforward(&mb)
				n.CalculateErrorInOutputLayer(x.OutputActivations, &mb)
				n.BackpropagateError(&mb)
			}
			dw, db := n.CalculateDerivatives(mbs)
			n.UpdateNetwork(eta, dw, db)
		}
	}
}
