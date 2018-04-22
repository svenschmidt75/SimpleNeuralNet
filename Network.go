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

	// L2 regularization
	Lambda float64
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
	err = encoder.Encode(n.Lambda)
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
	err = decoder.Decode(&n.weights)
	if err != nil {
		return err
	}
	return decoder.Decode(&n.Lambda)
	return err
}

func CreateNetwork(layers []int, lambda float64) Network {
	nBiases := sum(layers[1:])
	return Network{nodes: layers, biases: make([]float64, nBiases), weights: make([]float64, nWeights(layers)), Lambda: lambda}
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

func (n Network) GetActivation(index int, layer int, mb *Minibatch) float64 {
	aIdx := n.GetNodeIndex(index, layer)
	return mb.a[aIdx]
}

func (n *Network) SetActivation(a float64, index int, layer int, mb *Minibatch) {
	aIdx := n.GetNodeIndex(index, layer)
	mb.a[aIdx] = a
}

func (n Network) getDeltaBaseIndex(layer int) int {
	return n.getNodeBaseIndex(layer)
}

func (n Network) getDeltaIndex(index int, layer int) int {
	return n.GetNodeIndex(index, layer)
}

func (n Network) GetDelta(index int, layer int, mb *Minibatch) float64 {
	deltaIdx := n.getDeltaIndex(index, layer)
	return mb.delta[deltaIdx]
}

func (n *Network) SetDelta(delta float64, index int, layer int, mb *Minibatch) {
	deltaIdx := n.getDeltaIndex(index, layer)
	mb.delta[deltaIdx] = delta
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

func (n Network) GetBias(index int, layer int) float64 {
	biasIndex := n.GetBiasIndex(index, layer)
	return n.biases[biasIndex]
}

func (n *Network) SetBias(b float64, index int, layer int) {
	biasIndex := n.GetBiasIndex(index, layer)
	n.biases[biasIndex] = b
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

func (n Network) GetWeight(i int, j int, layer int) float64 {
	weightIndex := n.GetWeightIndex(i, j, layer)
	return n.weights[weightIndex]
}

func (n *Network) SetWeight(w float64, i int, j int, layer int) {
	weightIndex := n.GetWeightIndex(i, j, layer)
	n.weights[weightIndex] = w
}

func Sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

func SigmoidPrime(z float64) float64 {
	// derivative of the sigmoid function
	s := Sigmoid(z)
	return s * (1.0 - s)
}

func (n *Network) CalculateZ(i int, layer int, mb *Minibatch) float64 {
	var z float64
	nPrevLayer := n.nNodesInLayer(layer - 1)
	for j := 0; j < nPrevLayer; j++ {
		a_j := n.GetActivation(j, layer-1, mb)
		w_ij := n.GetWeight(i, j, layer)
		z += w_ij * a_j
	}
	b := n.GetBias(i, layer)
	z += b
	return z
}

func (n *Network) FeedforwardActivation(i int, layer int, mb *Minibatch) float64 {
	if l := len(n.nodes); layer == 0 || layer >= l {
		panic(fmt.Sprintf("Node layer index=%v must be bigger than 0 and smaller than the number of layers=%v", layer, l))
	}
	z := n.CalculateZ(i, layer, mb)
	a := Sigmoid(z)
	return a
}

func (n *Network) FeedforwardLayer(layer int, mb *Minibatch) {
	if layer == 0 {
		return
	}
	nLayer := n.nNodesInLayer(layer)
	for i := 0; i < nLayer; i++ {
		a := n.FeedforwardActivation(i, layer, mb)
		n.SetActivation(a, i, layer, mb)
	}
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
		nWeights := n.nWeights()
		for widx := 0; widx < nWeights; widx++ {
			n.weights[widx] = rand.Float64() / 100.0
		}
		nBiases := n.nBiases()
		for bidx := 0; bidx < nBiases; bidx++ {
			n.biases[bidx] = rand.Float64() / 100.0
		}
	}
}

func (n *Network) SetInputActivations(inputActivations []float64, mb *Minibatch) {
	if len(inputActivations) != n.nodes[0] {
		panic(fmt.Sprintf("Input activation size %v does not match number of activations %v in input layer", len(inputActivations), n.nodes[0]))
	}
	for idx, a := range inputActivations {
		n.SetActivation(a, idx, 0, mb)
	}
}

func (n *Network) BackpropagateError(mb *Minibatch) {
	// Equation (45), Chapter 2 of http://neuralnetworksanddeeplearning.com
	outputLayerIdx := n.getOutputLayerIndex()
	for layer := outputLayerIdx - 1; layer > 0; layer-- {
		nNodes := n.nNodesInLayer(layer)
		nNextNodes := n.nNodesInLayer(layer + 1)
		for j := 0; j < nNodes; j++ {
			var tmp float64
			for k := 0; k < nNextNodes; k++ {
				weight_kj := n.GetWeight(k, j, layer+1)
				delta_k := n.GetDelta(k, layer+1, mb)
				tmp += weight_kj * delta_k
			}
			z_j := n.CalculateZ(j, layer, mb)
			s := SigmoidPrime(z_j)
			error := tmp * s
			n.SetDelta(error, j, layer, mb)
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
	db := make([]float64, n.nNodes())

	nMiniBatches := len(mbs)
	for layer := range n.nodes {
		if layer == 0 {
			continue
		}
		nNodes := n.nNodesInLayer(layer)
		nPrevNodes := n.nNodesInLayer(layer - 1)
		for j := 0; j < nNodes; j++ {
			for k := 0; k < nPrevNodes; k++ {
				// w_jk^l
				var dw_jk float64
				for mbIdx := range mbs {
					mb := mbs[mbIdx]
					a := n.GetActivation(k, layer-1, &mb)
					delta := n.GetDelta(j, layer, &mb)
					dCx_dw := a * delta
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
				delta := n.GetDelta(j, layer, &mb)
				db_j += delta
			}
			db_j /= float64(nMiniBatches)
			bIdx := n.GetBiasIndex(j, layer)
			db[bIdx] = db_j
		}
	}
	return dw, db
}

func (n *Network) UpdateNetwork(eta float32, dw []float64, db []float64, nTrainingSamples int) {
	for layer := range n.nodes {
		if layer == 0 {
			continue
		}
		nNodes := n.nNodesInLayer(layer)
		nPrevNodes := n.nNodesInLayer(layer - 1)
		for j := 0; j < nNodes; j++ {
			for k := 0; k < nPrevNodes; k++ {
				// w_jk^l
				wIdx := n.GetWeightIndex(j, k, layer)
				dw_jk := dw[wIdx]
				w_jk := n.GetWeight(j, k, layer)
				update := (1-float64(eta)*n.Lambda/float64(nTrainingSamples))*w_jk - float64(eta)*dw_jk
				n.SetWeight(update, j, k, layer)
			}
			// b_j^l
			bIdx := n.GetBiasIndex(j, layer)
			db_j := db[bIdx]
			b_j := n.GetBias(j, layer)
			b_j -= float64(eta) * db_j
			n.SetBias(b_j, j, layer)
		}
	}
}

func (n *Network) Train(trainingSamples []MNISTImport.TrainingSample, validationSamples []MNISTImport.TrainingSample, epochs int, eta float32, miniBatchSize int, costFunction CostFunction) {
	// Stochastic Gradient Decent
	sizeMiniBatch := min(len(trainingSamples), miniBatchSize)
	nMiniBatches := len(trainingSamples) / sizeMiniBatch
	mbs := CreateMiniBatches(sizeMiniBatch, n.nNodes(), n.nWeights())

	fmt.Printf("\nTraining batch size: %d\n", len(trainingSamples))
	fmt.Printf("Validation batch size: %d\n", len(validationSamples))
	fmt.Printf("Minibatch size: %d\n", sizeMiniBatch)
	fmt.Printf("Number of minibatches: %d\n", nMiniBatches)
	fmt.Printf("Learning rate: %f\n", eta)
	fmt.Printf("Cost function: %s\n", costFunction)
	fmt.Printf("L2 regularization: %f\n\n", n.Lambda)

	var innerLoop = func(maxIndex int, offset int, indices []int) {
		for i := 0; i < maxIndex; i++ {
			mb := mbs[i]
			index := indices[offset*sizeMiniBatch+i]
			x := trainingSamples[index]
			n.SetInputActivations(x.InputActivations, &mb)
			n.Feedforward(&mb)
			costFunction.CalculateErrorInOutputLayer(n, x.OutputActivations, &mb)
			n.BackpropagateError(&mb)
		}
		dw, db := n.CalculateDerivatives(mbs)

		var gradCNorm float64
		for _, dwi := range dw {
			gradCNorm += dwi * dwi
			//			fmt.Printf("dwi = %f\n", dwi*dwi)
		}
		for _, dbi := range db {
			gradCNorm += dbi * dbi
			//			fmt.Printf("dbi = %f\n", dbi*dbi)
		}
		//		fmt.Printf("|gradC| = %f\n", math.Sqrt(gradCNorm))

		n.UpdateNetwork(eta, dw, db, len(trainingSamples))
	}

	for epoch := 0; epoch < epochs; epoch++ {
		indices := GenerateRandomIndices(len(trainingSamples))
		for j := 0; j < nMiniBatches; j++ {
			//			fmt.Printf("Minibatch %d of %d...\n", j, nMiniBatches)
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
		cost := costFunction.Evaluate(n, trainingSamples)
		output += fmt.Sprintf(" - cost %f\n", cost)
		fmt.Print(output)
	}
}

func (n *Network) GetOutputLayerActivations(mb *Minibatch) []float64 {
	idx := n.getNodeBaseIndex(n.getOutputLayerIndex())
	as := mb.a[idx:]
	return as
}

func (n *Network) RunSamples(trainingSamples []MNISTImport.TrainingSample, showFailures bool) float32 {
	var correctPredictions int
	mb := CreateMiniBatch(n.nNodes(), n.nWeights())
	for testIdx := range trainingSamples {
		n.SetInputActivations(trainingSamples[testIdx].InputActivations, &mb)
		n.Feedforward(&mb)
		as := n.GetOutputLayerActivations(&mb)
		predictionClass := GetClass(as)
		expectedClass := GetClass(trainingSamples[testIdx].OutputActivations)
		if expectedClass == predictionClass {
			correctPredictions++
		} else if showFailures {
			fmt.Printf("Image %d: is %d, classified as %d\n", testIdx, expectedClass, predictionClass)
		}
	}
	accuracy := float32(correctPredictions) / float32(len(trainingSamples))
	return accuracy
}
