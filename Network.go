package main

import (
	"SimpleNeuralNet/LinAlg"
	"SimpleNeuralNet/MNISTImport"
	"bytes"
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"time"
)

// weights: The weights w_ij^{l} are ordered by layer l, and for each layer,
// by i, then j.
// Example: w_00^{1}, w_01^{1}, ..., w_0m^{1}, w_10^{1}, ..., w_1m^{1}, ..., w_n0^{1}, ..., w_nm^{1},
//          w_00^{2}, w_01^{2}, ..., w_0m^{2}, w_10^{2}, ..., w_1m^{2}, ..., w_n0^{2}, ..., w_nm^{2},
type Network struct {
	// how many activations per layer
	nodes []int

	// bias vectors, ordered by layer
	biases []LinAlg.Vector

	// Weight matrices. w_{ij}^l connects a_i^l with a_j^{l-1}
	weights []LinAlg.Matrix
}

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

//
// Implement interface 'GobEncoder'
//
func (n *Network) GobEncode() ([]byte, error) {
	w := new(bytes.Buffer)
	encoder := gob.NewEncoder(w)
	err := encoder.Encode(n.nodes)
	if err != nil {
		return nil, err
	}
	err = encoder.Encode(n.biases)
	if err != nil {
		return nil, err
	}
	err = encoder.Encode(n.weights)
	if err != nil {
		return nil, err
	}
	return w.Bytes(), nil
}

//
// Implement interface 'GobDecoder'
//
func (n *Network) GobDecode(buf []byte) error {
	r := bytes.NewBuffer(buf)
	decoder := gob.NewDecoder(r)
	err := decoder.Decode(&n.nodes)
	if err != nil {
		return err
	}
	err = decoder.Decode(&n.biases)
	if err != nil {
		return err
	}
	return decoder.Decode(&n.weights)
}

func CreateNetwork(layers []int) Network {
	return Network{nodes: layers, biases: createBiasVector(layers), weights: createWeightMatrices(layers)}
}

func createWeightMatrices(layers []int) []LinAlg.Matrix {
	result := make([]LinAlg.Matrix, len(layers))
	for idx := 1; idx < len(layers); idx++ {
		rows := layers[idx]
		cols := layers[idx-1]
		result[idx] = *LinAlg.MakeEmptyMatrix(rows, cols)
	}
	return result
}

func createBiasVector(layers []int) []LinAlg.Vector {
	result := make([]LinAlg.Vector, len(layers))
	for idx, nNodes := range layers[1:] {
		result[idx] = *LinAlg.MakeEmptyVector(nNodes)
	}
	return result
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

func (n *Network) GetLayers() []int {
	return n.nodes
}

func (n Network) nWeights() int {
	return nWeights(n.nodes)
}

func (n Network) nWeightsInLayer(layer int) int {
	if layer == 0 {
		panic("Layer 0 has no weights")
	}
	idx1 := nWeights(n.nodes[0:layer])
	idx2 := nWeights(n.nodes[0 : layer+1])
	return idx2 - idx1
}

func (n Network) getOutputLayerIndex() int {
	return len(n.nodes) - 1
}

func (n Network) nNodes() int {
	return sum(n.nodes)
}

func (n Network) nNodesInLayer(layer int) int {
	if l := len(n.nodes); layer >= l {
		panic(fmt.Sprintf("Layer index %v must be <= %v", layer, l))
	}
	return sum(n.nodes[layer : layer+1])
}

func (n Network) getNodeBaseIndex(layer int) int {
	bi := sum(n.nodes[0:layer])
	return bi
}

func (n Network) GetNodeIndex(index int, layer int) int {
	if layer >= len(n.nodes) {
		panic(fmt.Sprintf("Layer index=%v must be smaller than the number of nodes=%v", layer, len(n.nodes)))
	}
	if index >= n.nodes[layer] {
		panic(fmt.Sprintf("Node index i=%v must be smaller than the number of nodes=%v in layer %v", index, n.nodes[layer], layer))
	}
	bi := n.getNodeBaseIndex(layer)
	return bi + index
}

func (n *Network) GetActivation(layer int, mb *Minibatch) LinAlg.Vector {
	return mb.a[layer]
}

func (n Network) getDeltaBaseIndex(layer int) int {
	return n.getNodeBaseIndex(layer)
}

func (n Network) getDeltaIndex(index int, layer int) int {
	return n.GetNodeIndex(index, layer)
}

func (n *Network) GetDelta(layer int, mb *Minibatch) LinAlg.Vector {
	return mb.delta[layer]
}

func (n *Network) SetDelta(layer int, delta LinAlg.Vector, mb *Minibatch) {
	mb.delta[layer] = delta
}

func (n Network) nBiases() int {
	return sum(n.nodes[1:])
}

func (n *Network) getBiasBaseIndex(layer int) int {
	bi := sum(n.nodes[1:layer])
	return bi
}

func (n *Network) GetBiasIndex(index int, layer int) int {
	if layer >= len(n.nodes) {
		panic(fmt.Sprintf("Bias layer index=%v must be smaller than the number of nodes=%v", layer, len(n.nodes)))
	}
	if index >= n.nodes[layer] {
		panic(fmt.Sprintf("Bias index i=%v must be smaller than the number of activations=%v in layer %v", index, n.nodes[layer], layer))
	}
	bi := n.getBiasBaseIndex(layer)
	return bi + index
}

func (n *Network) GetBias(layer int) *LinAlg.Vector {
	return &n.biases[layer]
}

func (n *Network) SetBias(layer int, b LinAlg.Vector) {
	n.biases[layer] = b
}

func (n *Network) GetWeights(layer int) *LinAlg.Matrix {
	return &n.weights[layer]
}

func (n *Network) SetWeights(layer int, w LinAlg.Matrix) {
	n.weights[layer] = w
}

// Start index of w^{l}_ij, i.e. linear index of w^{layer}_00 in
// n.weights
func (n Network) getWeightBaseIndex(layer int) int {
	return nWeights(n.nodes[0:layer])
}

func (n Network) GetWeightIndex(i int, j int, layer int) int {
	// Remember the meaning of the indices: w_ij^{l) is the weight from
	// neuron a_j^{l-1} to neuron a_i^{l}.
	if layer == 0 {
		panic(fmt.Sprintf("Weight layer index=%v must be bigger than 0 and smaller than the number of nodes=%v", layer, len(n.nodes)))
	}
	if layer >= len(n.nodes) {
		panic(fmt.Sprintf("Weight layer index=%v must be smaller than the number of nodes=%v", layer, len(n.nodes)))
	}
	if i >= n.nodes[layer] {
		panic(fmt.Sprintf("Weight index i=%v must be smaller than the number of activations=%v in layer %v", i, n.nodes[layer], layer))
	}
	if j >= n.nodes[layer-1] {
		panic(fmt.Sprintf("Weight index j=%v must be smaller than the number of activations=%v in layer %v", j, n.nodes[layer-1], layer-1))
	}
	bi := n.getWeightBaseIndex(layer)
	nl1 := n.nNodesInLayer(layer - 1)
	bi = bi + i*nl1
	return bi + j
}

func Sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

func SigmoidPrime(z float64) float64 {
	// derivative of the sigmoid function
	s := Sigmoid(z)
	return s * (1.0 - s)
}

func (n *Network) CalculateZ(layer int, mb *Minibatch) {
	w := n.weights[layer]
	a := mb.a[layer-1]
	b := n.biases[layer]
	wa := w.Ax(&a)
	z := LinAlg.AddVectors(wa, &b)
	mb.z[layer] = *z
}

func (n *Network) FeedforwardLayer(layer int, mb *Minibatch) {
	n.CalculateZ(layer, mb)
	mb.a[layer] = *mb.z[layer].F(Sigmoid)
}

func (n *Network) Feedforward(mb *Minibatch) {
	for layer := range n.nodes {
		if layer == 0 {
			continue
		}
		n.FeedforwardLayer(layer, mb)
	}
}

func (n *Network) InitializeNetworkWeightsAndBiases() {
	for layer := range n.nodes {
		if layer == 0 {
			continue
		}
		w := n.weights[layer]
		for row := 0; row < w.Rows; row++ {
			for col := 0; col < w.Cols; col++ {
				w.Set(row, col, rand.Float64()/100.0)
			}
		}
		b := n.biases[layer]
		for row := 0; row < b.Size(); row++ {
			b.Set(row, rand.Float64()/100.0)
		}
	}
}

func (n *Network) BackpropagateError(mb *Minibatch) {
	// Equation (45), Chapter 2 of http://neuralnetworksanddeeplearning.com
	outputLayerIdx := n.getOutputLayerIndex()
	for layer := outputLayerIdx - 1; layer > 0; layer-- {
		delta_next := mb.delta[layer+1]
		s := mb.z[layer].F(SigmoidPrime)
		delta := n.GetWeights(layer + 1).Transpose().Ax(&delta_next).Hadamard(s)
		mb.delta[layer] = *delta
	}
}

func (n *Network) CalculateDerivatives(mbs []Minibatch) ([]LinAlg.Matrix, []LinAlg.Vector) {
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
	dw := make([]LinAlg.Matrix, n.getOutputLayerIndex()+1)

	// d C_x / d_bj^l
	db := make([]LinAlg.Vector, n.getOutputLayerIndex()+1)

	nMiniBatches := len(mbs)
	for layer := range n.nodes {
		if layer == 0 {
			continue
		}

		i := n.nodes[layer]
		dCdw := LinAlg.MakeEmptyMatrix(i, i)
		dCdb := LinAlg.MakeEmptyVector(i)
		for mbIdx := range mbs {
			mb := mbs[mbIdx]
			delta := mb.delta[layer]
			tmp := LinAlg.OuterProduct(&mb.a[layer-1], &delta)
			dCdw.Add(tmp)
			dCdb.Add(&delta)
		}
		dCdw.Scalar(1 / float64(nMiniBatches))
		dw[layer] = *dCdw
		dCdb.Scalar(1 / float64(nMiniBatches))
		db[layer] = *dCdb
	}
	return dw, db
}

func (n *Network) UpdateNetwork(eta float32, lambda float64, dw []LinAlg.Matrix, db []LinAlg.Vector, nTrainingSamples int) {
	for layer := range n.nodes {
		if layer == 0 {
			continue
		}
		w := n.GetWeights(layer)
		w.Scalar(1 - float64(eta)*lambda/float64(nTrainingSamples))
		tmp2 := dw[layer]
		tmp2.Scalar(float64(eta))
		w.Sub(&tmp2)
		n.SetWeights(layer, *w)

		b := n.GetBias(layer)
		tmp3 := db[layer]
		tmp3.Scalar(-float64(eta))
		LinAlg.SubtractVectors(b, &tmp3)
		n.SetBias(layer, *b)
	}
}

func (n *Network) Train(trainingSamples []MNISTImport.TrainingSample, validationSamples []MNISTImport.TrainingSample, epochs int, eta float32, lambda float64, miniBatchSize int, costFunction CostFunction) {
	// Stochastic Gradient Decent
	sizeMiniBatch := min(len(trainingSamples), miniBatchSize)
	nMiniBatches := len(trainingSamples) / sizeMiniBatch
	mbs := CreateMiniBatches(sizeMiniBatch, n.GetLayers())

	fmt.Printf("\nTraining batch size: %d\n", len(trainingSamples))
	fmt.Printf("Validation batch size: %d\n", len(validationSamples))
	fmt.Printf("Minibatch size: %d\n", sizeMiniBatch)
	fmt.Printf("Number of minibatches: %d\n", nMiniBatches)
	fmt.Printf("Learning rate: %f\n", eta)
	fmt.Printf("Cost function: %s\n", costFunction)
	fmt.Printf("L2 regularization: %f\n\n", lambda)

	var innerLoop = func(maxIndex int, offset int, indices []int) {
		for i := 0; i < maxIndex; i++ {
			mb := mbs[i]
			index := indices[offset*sizeMiniBatch+i]
			x := trainingSamples[index]
			mb.a[0] = x.InputActivations
			n.Feedforward(&mb)
			costFunction.CalculateErrorInOutputLayer(n, &x.OutputActivations, &mb)
			n.BackpropagateError(&mb)
		}
		dw, db := n.CalculateDerivatives(mbs)
		n.UpdateNetwork(eta, lambda, dw, db, len(trainingSamples))
	}

	for epoch := 0; epoch < epochs; epoch++ {
		indices := GenerateRandomIndices(len(trainingSamples))
		for j := 0; j < nMiniBatches; j++ {
			innerLoop(sizeMiniBatch, j, indices)
		}
		if remainder := len(trainingSamples) - sizeMiniBatch*nMiniBatches; remainder > 0 {
			innerLoop(remainder, nMiniBatches, indices)
		}
		output := fmt.Sprintf("Epoch %d", epoch+1)
		accuracy := n.RunSamples(trainingSamples, false)
		output += fmt.Sprintf(" - training accuracy %f", accuracy)
		if len(validationSamples) > 0 {
			accuracy := n.RunSamples(validationSamples, false)
			output += fmt.Sprintf(" - validation accuracy %f", accuracy)
		}
		cost := costFunction.Evaluate(n, lambda, trainingSamples)
		output += fmt.Sprintf(" - cost %f\n", cost)
		fmt.Print(output)
	}
}

func (n *Network) GetOutputLayerActivations(mb *Minibatch) *LinAlg.Vector {
	idx := n.getOutputLayerIndex()
	return &mb.a[idx]
}

func (n *Network) RunSamples(trainingSamples []MNISTImport.TrainingSample, showFailures bool) float32 {
	var correctPredictions int
	mb := CreateMiniBatch(n.nodes)
	for testIdx := range trainingSamples {
		mb.a[0] = trainingSamples[testIdx].InputActivations
		n.Feedforward(&mb)
		predictionClass := GetClass(n.GetOutputLayerActivations(&mb))
		expectedClass := GetClass(&trainingSamples[testIdx].OutputActivations)
		if expectedClass == predictionClass {
			correctPredictions++
		} else if showFailures {
			fmt.Printf("Image %d: is %d, classified as %d\n", testIdx, expectedClass, predictionClass)
		}
	}
	accuracy := float32(correctPredictions) / float32(len(trainingSamples))
	return accuracy
}
