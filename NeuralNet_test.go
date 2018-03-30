package main

import (
	"SimpleNeuralNet/MNISTImport"
	"fmt"
	"math"
	"testing"
)

func TestSigmoid(t *testing.T) {
	if Sigmoid(0) != 0.5 {
		// Assert
		t.Error("Unexpected result")
	}
}

func TestNumberOfWeights(t *testing.T) {
	tables := []struct {
		xs       []int
		nBiases  int
		nWeights int
	}{
		{[]int{1, 1, 1}, 2, 2},
		{[]int{2, 1, 2}, 3, 4},
		{[]int{2, 3, 2}, 5, 12},
	}

	for _, item := range tables {
		network := CreateNetwork(item.xs)
		nWeights := len(network.weights)
		if nWeights != item.nWeights {
			t.Errorf("Expected 2, but is %v", nWeights)
		}
	}
}

func TestNumberOfBiases(t *testing.T) {
	tables := []struct {
		xs       []int
		nBiases  int
		nWeights int
	}{
		{[]int{1, 1, 1}, 2, 2},
		{[]int{2, 1, 2}, 3, 4},
		{[]int{2, 3, 2}, 5, 12},
	}

	for _, item := range tables {
		network := CreateNetwork(item.xs)
		nBiases := len(network.biases)
		if nBiases != item.nBiases {
			t.Errorf("Expected 2, but is %v", nBiases)
		}
	}
}

func TestActivationIndex(t *testing.T) {
	type Activation struct {
		i     int
		layer int
		index int
	}

	tables := []struct {
		xs       []int
		nBiases  int
		nWeights int
		ais      []Activation
	}{
		{[]int{2, 3, 2}, 5, 12, []Activation{
			{0, 1, 2},
			{1, 2, 6},
		}},
	}

	for _, ts := range tables {
		network := CreateNetwork(ts.xs)
		for _, as := range ts.ais {
			ai := network.GetNodeIndex(as.i, as.layer)
			if ai != as.index {
				t.Errorf("Expected %v, but is %v", as.index, ai)
			}
		}
	}
}

func TestBiasIndex(t *testing.T) {
	type Bias struct {
		i     int
		layer int
		index int
	}

	tables := []struct {
		xs       []int
		nBiases  int
		nWeights int
		ais      []Bias
	}{
		{[]int{2, 3, 2}, 5, 12, []Bias{
			{1, 1, 1},
			{0, 2, 3},
		}},
	}

	for _, ts := range tables {
		network := CreateNetwork(ts.xs)
		for _, as := range ts.ais {
			ai := network.GetBiasIndex(as.i, as.layer)
			if ai != as.index {
				t.Errorf("Expected %v, but is %v", as.index, ai)
			}
		}
	}
}

func TestWeightIndex(t *testing.T) {
	type Weight struct {
		i     int
		j     int
		layer int
		index int
	}

	tables := []struct {
		xs       []int
		nBiases  int
		nWeights int
		ais      []Weight
	}{
		{[]int{2, 3, 2}, 5, 12, []Weight{
			{0, 0, 1, 0},
			{0, 1, 1, 1},
			{1, 0, 1, 2},
			{1, 1, 1, 3},
			{2, 0, 1, 4},
			{2, 1, 1, 5},
			{0, 0, 2, 6},
			{0, 1, 2, 7},
			{0, 2, 2, 8},
			{1, 0, 2, 9},
			{1, 1, 2, 10},
			{1, 2, 2, 11},
		}},
	}

	for _, ts := range tables {
		network := CreateNetwork(ts.xs)
		for _, as := range ts.ais {
			ai := network.GetWeightIndex(as.i, as.j, as.layer)
			if ai != as.index {
				t.Errorf("Expected %v, but is %v", as.index, ai)
			}
		}
	}
}

func TestNumberOfWeightsPerLayer(t *testing.T) {
	type Layer struct {
		layer    int
		expected int
	}

	tables := []struct {
		xs  []int
		ais []Layer
	}{
		{[]int{2, 3, 2}, []Layer{
			{1, 6},
			{2, 6},
		}},
		{[]int{28 * 28, 100, 10}, []Layer{
			{1, 100 * 28 * 28},
			{2, 10 * 100},
		}},
	}

	for _, ts := range tables {
		network := CreateNetwork(ts.xs)
		for _, as := range ts.ais {
			ai := network.nWeightsInLayer(as.layer)
			if ai != as.expected {
				t.Errorf("Expected %v weights in layer %v, but is %v", as.expected, as.layer, ai)
			}
		}
	}
}

func CreateTestNetwork() (Network, Minibatch) {
	network := CreateNetwork([]int{2, 3, 2})
	network.weights[network.GetWeightIndex(0, 0, 1)] = 1
	network.weights[network.GetWeightIndex(0, 1, 1)] = 2
	network.weights[network.GetWeightIndex(1, 0, 1)] = 3
	network.weights[network.GetWeightIndex(1, 1, 1)] = 4
	network.weights[network.GetWeightIndex(2, 0, 1)] = 5
	network.weights[network.GetWeightIndex(2, 1, 1)] = 6
	network.weights[network.GetWeightIndex(0, 0, 2)] = 7
	network.weights[network.GetWeightIndex(0, 1, 2)] = 8
	network.weights[network.GetWeightIndex(0, 2, 2)] = 9
	network.weights[network.GetWeightIndex(1, 0, 2)] = 10
	network.weights[network.GetWeightIndex(1, 1, 2)] = 11
	network.weights[network.GetWeightIndex(1, 2, 2)] = 12

	network.biases[network.GetBiasIndex(0, 1)] = 1
	network.biases[network.GetBiasIndex(1, 1)] = 2
	network.biases[network.GetBiasIndex(2, 1)] = 3
	network.biases[network.GetBiasIndex(0, 2)] = 4
	network.biases[network.GetBiasIndex(1, 2)] = 5

	mb := CreateMiniBatch(7, 12)
	mb.a[network.GetNodeIndex(0, 0)] = 1
	mb.a[network.GetNodeIndex(1, 0)] = 2

	return network, mb
}

func CreateTestNetwork2() Network {
	network := CreateNetwork([]int{2, 3, 2})
	network.weights[network.GetWeightIndex(0, 0, 1)] = 0.03645
	network.weights[network.GetWeightIndex(0, 1, 1)] = 0.3645
	network.weights[network.GetWeightIndex(1, 0, 1)] = 0.14352
	network.weights[network.GetWeightIndex(1, 1, 1)] = 0.03645
	network.weights[network.GetWeightIndex(2, 0, 1)] = 0.028346
	network.weights[network.GetWeightIndex(2, 1, 1)] = 0.5363
	network.weights[network.GetWeightIndex(0, 0, 2)] = 0.2534
	network.weights[network.GetWeightIndex(0, 1, 2)] = 0.4132
	network.weights[network.GetWeightIndex(0, 2, 2)] = 0.823746
	network.weights[network.GetWeightIndex(1, 0, 2)] = 0.0374
	network.weights[network.GetWeightIndex(1, 1, 2)] = 0.6153
	network.weights[network.GetWeightIndex(1, 2, 2)] = 0.243

	network.biases[network.GetBiasIndex(0, 1)] = 0.23
	network.biases[network.GetBiasIndex(1, 1)] = 0.2635
	network.biases[network.GetBiasIndex(2, 1)] = 0.03756
	network.biases[network.GetBiasIndex(0, 2)] = 0.3746
	network.biases[network.GetBiasIndex(1, 2)] = 0.063

	return network
}

func TestCalculateZ(t *testing.T) {
	network, mb := CreateTestNetwork()

	tables := []struct {
		layer int
		index int
		value float64
	}{
		{1, 0, 6.0},
		{1, 1, 13.0},
		{1, 2, 20.0},
		{2, 0, 330.0},
		{2, 1, 448.0},
	}

	for _, ts := range tables {
		v := network.CalculateZ(ts.index, ts.layer, &mb)

		// for this test, we assume the activation function is the identity
		network.SetActivation(v, ts.index, ts.layer, &mb)
		if v != ts.value {
			t.Errorf("Expected %v, but is %v", ts.value, v)
		}
	}
}

var EPSILON = 0.00000001

func floatEquals(a, b float64) bool {
	return math.Abs(a-b) < EPSILON
}

func TestFeedforwardActivation(t *testing.T) {
	network, mb := CreateTestNetwork()

	tables := []struct {
		layer int
		index int
		value float64
	}{
		{1, 0, Sigmoid(6.0)},
		{1, 1, Sigmoid(13.0)},
		{1, 2, Sigmoid(20.0)},
		{2, 0, Sigmoid(330.0)},
		{2, 1, Sigmoid(448.0)},
	}

	for _, ts := range tables {
		v := network.FeedforwardActivation(ts.index, ts.layer, &mb)
		network.SetActivation(v, ts.index, ts.layer, &mb)
		if floatEquals(v, ts.value) == false {
			t.Errorf("Expected %v, but is %v", ts.value, v)
		}
	}
}

func TestFeedforward(t *testing.T) {
	network, mb := CreateTestNetwork()
	network.Feedforward(&mb)

	tables := []struct {
		layer int
		index int
		value float64
	}{
		{1, 0, Sigmoid(6.0)},
		{1, 1, Sigmoid(13.0)},
		{1, 2, Sigmoid(20.0)},
		{2, 0, Sigmoid(330.0)},
		{2, 1, Sigmoid(448.0)},
	}

	for _, ts := range tables {
		v := network.GetActivation(ts.index, ts.layer, &mb)
		if floatEquals(v, ts.value) == false {
			t.Errorf("Expected %v, but is %v", ts.value, v)
		}
	}
}

func TestSetInputActivations(t *testing.T) {
	network, mb := CreateTestNetwork()

	network.SetInputActivations([]float64{4.9, 3.2}, &mb)

	if a := network.GetActivation(0, 0, &mb); floatEquals(4.9, a) == false {
		t.Errorf("Expected 4.9, but is %v", a)
	}

	if a := network.GetActivation(1, 0, &mb); floatEquals(3.2, a) == false {
		t.Errorf("Expected 3.2, but is %v", a)
	}
}

func TestCalculateErrorInOutputLayer(t *testing.T) {
	network := CreateTestNetwork2()
	mb := CreateMiniBatch(7, 12)
	network.SetActivation(1, 0, 1, &mb)
	network.SetActivation(2, 1, 1, &mb)
	network.SetActivation(3, 2, 1, &mb)
	outputLayerIdx := 2

	tables := []struct {
		outputActivations []float64
		expectedClass     int
		error             []float64
	}{
		{[]float64{1, 1}, 1, []float64{0, 0}},
		{[]float64{0, 0}, 1, []float64{-0.01897348347524417, -0.10026647037539357}},
		{[]float64{0.24, 0.21}, 1, []float64{-0.014419847441185569, -0.07921051159656092}},
	}

	for _, ts := range tables {
		network.SetActivation(ts.outputActivations[0], 0, outputLayerIdx, &mb)
		network.SetActivation(ts.outputActivations[1], 1, outputLayerIdx, &mb)
		network.CalculateErrorInOutputLayer(ts.expectedClass, &mb)
		if nabla := network.GetNabla(0, 2, &mb); floatEquals(ts.error[0], nabla) == false {
			t.Errorf("Expected %v, but was %v", ts.error[0], nabla)
		}
		if nabla := network.GetNabla(1, 2, &mb); floatEquals(ts.error[1], nabla) == false {
			t.Errorf("Expected %v, but was %v", ts.error[1], nabla)
		}
	}
}

func TestBackpropagateError(t *testing.T) {
	network := CreateTestNetwork2()
	mb := CreateMiniBatch(7, 12)
	network.SetActivation(0.32, 0, 0, &mb)
	network.SetActivation(0.56, 1, 0, &mb)
	network.Feedforward(&mb)
	network.CalculateErrorInOutputLayer(0, &mb)
	network.BackpropagateError(&mb)

	if nabla := network.GetNabla(0, 1, &mb); floatEquals(0.007357185735347161, nabla) == false {
		t.Errorf("Expected %v, but was %v", 0.007357185735347161, nabla)
	}
	if nabla := network.GetNabla(1, 1, &mb); floatEquals(0.016680036100361194, nabla) == false {
		t.Errorf("Expected %v, but was %v", 0.016680036100361194, nabla)
	}
	if nabla := network.GetNabla(2, 1, &mb); floatEquals(0.025347427582781648, nabla) == false {
		t.Errorf("Expected %v, but was %v", 0.025347427582781648, nabla)
	}
}

func TestCalculateDerivatives(t *testing.T) {
	network, mb := CreateTestNetwork()
	network.Feedforward(&mb)
	network.CalculateErrorInOutputLayer(1, &mb)
	network.BackpropagateError(&mb)

	dw, db := network.CalculateDerivatives([]Minibatch{mb})

	if wIdx := network.GetWeightIndex(0, 0, 1); floatEquals(0, dw[wIdx]) == false {
		t.Errorf("dw(%v, %v), layer %v, does not equal to %v, but instead %v", 0, 0, 1, 0, dw[wIdx])
	}
	if bIdx := network.GetBiasIndex(0, 1); floatEquals(0, db[bIdx]) == false {
		t.Errorf("db(%v), layer %v, does not equal to %v, but instead %v", 0, 0, 0, db[bIdx])
	}
}

func TestTrain(t *testing.T) {
	network, _ := CreateTestNetwork()

	ts := []MNISTImport.TrainingSample{MNISTImport.CreateTrainingSample([]float64{0.34, 0.43}, 1), MNISTImport.CreateTrainingSample([]float64{0.14, 0.03}, 1)}
	network.Train(ts, []MNISTImport.TrainingSample{}, 2, 0.001, 10)

	mb := CreateMiniBatch(12, 7)
	network.SetInputActivations([]float64{0.34, 0.43}, &mb)
	network.Feedforward(&mb)

	// Assert
	if a := network.GetActivation(0, 2, &mb); floatEquals(0.34, a) == false {
		t.Errorf("Network gave wrong answer. Expected %v, was %v", 0.34, a)
	}
	if a := network.GetActivation(1, 2, &mb); floatEquals(0.43, a) == false {
		t.Errorf("Network gave wrong answer. Expected %v, was %v", 0.43, a)
	}
}

func TestTrainWithMNIST(t *testing.T) {
	network := CreateNetwork([]int{28 * 28, 100, 10})
	network.InitializeNetworkWeightsAndBiases()

	trainingData := MNISTImport.ImportData("/home/svenschmidt75/Develop/Go/go/src/SimpleNeuralNet/test_data/", "train-images50.idx3-ubyte", "train-labels50.idx1-ubyte")
	ts := trainingData.GenerateTrainingSamples(trainingData.Length())
	network.Train(ts, []MNISTImport.TrainingSample{}, 2, 0.5, 10)

	// Assert
	testData := MNISTImport.ImportData("/home/svenschmidt75/Develop/Go/MNIST", "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
	ts2 := testData.GenerateTrainingSamples(100)

	mb := CreateMiniBatch(network.nNodes(), network.nWeights())
	for index := 0; index < len(ts2); index++ {
		network.SetInputActivations(ts2[index].InputActivations, &mb)
		network.Feedforward(&mb)
		as := network.GetOutputLayerActivations(&mb)
		fmt.Printf("should %d: %v\n", ts2[index].ExpectedClass, as)
	}
}

func TestSerialization(t *testing.T) {
}
