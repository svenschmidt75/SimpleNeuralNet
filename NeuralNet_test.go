package main

import (
	"SimpleNeuralNet/LinAlg"
	"SimpleNeuralNet/MNISTImport"
	"SimpleNeuralNet/Utility"
	"bytes"
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
	network.GetWeights(1).Set(0, 0, 1)
	network.GetWeights(1).Set(0, 1, 2)
	network.GetWeights(1).Set(1, 0, 3)
	network.GetWeights(1).Set(1, 1, 4)
	network.GetWeights(1).Set(2, 0, 5)
	network.GetWeights(1).Set(2, 1, 6)
	network.GetWeights(2).Set(0, 0, 7)
	network.GetWeights(2).Set(0, 1, 8)
	network.GetWeights(2).Set(0, 2, 9)
	network.GetWeights(2).Set(1, 0, 10)
	network.GetWeights(2).Set(1, 1, 11)
	network.GetWeights(2).Set(1, 2, 12)

	network.GetBias(1).Set(0, 1)
	network.GetBias(1).Set(1, 2)
	network.GetBias(1).Set(2, 3)
	network.GetBias(2).Set(0, 4)
	network.GetBias(2).Set(1, 5)

	mb := CreateMiniBatch([]int{7, 12})
	mb.a[0].Set(0, 1)
	mb.a[0].Set(1, 2)

	return network, mb
}

func CreateTestNetwork2() Network {
	network := CreateNetwork([]int{2, 3, 2})
	network.GetWeights(1).Set(0, 0, 0.03645)
	network.GetWeights(1).Set(0, 1, 0.3645)
	network.GetWeights(1).Set(1, 0, 0.14352)
	network.GetWeights(1).Set(1, 1, 0.03645)
	network.GetWeights(1).Set(2, 0, 0.028346)
	network.GetWeights(1).Set(2, 1, 0.5363)
	network.GetWeights(2).Set(0, 0, 0.2534)
	network.GetWeights(2).Set(0, 1, 0.4132)
	network.GetWeights(2).Set(0, 2, 0.823746)
	network.GetWeights(2).Set(1, 0, 0.0374)
	network.GetWeights(2).Set(1, 1, 0.6153)
	network.GetWeights(2).Set(1, 2, 0.243)

	network.GetBias(1).Set(0, 0.23)
	network.GetBias(1).Set(1, 0.2635)
	network.GetBias(1).Set(2, 0.03756)
	network.GetBias(2).Set(0, 0.3746)
	network.GetBias(2).Set(1, 0.063)

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
		network.CalculateZ(ts.layer, &mb)
		v := mb.z[ts.layer]

		// for this test, we assume the activation function is the identity
		network.SetActivation(v, ts.layer, &mb)
		if v.Get(ts.index) != ts.value {
			t.Errorf("Expected %v, but is %v", ts.value, v)
		}
	}
}

const EPSILON = 0.00000001

func floatEquals(a, b float64, eps float64) bool {
	return math.Abs(a-b) < eps
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
		network.FeedforwardLayer(ts.layer, &mb)
		a := network.GetActivation(ts.layer, &mb)
		if floatEquals(a.Get(ts.index), ts.value, EPSILON) == false {
			t.Errorf("Expected %v, but is %v", ts.value, a.Get(ts.index))
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
		v := network.GetActivation(ts.layer, &mb)
		if floatEquals(v.Get(ts.index), ts.value, EPSILON) == false {
			t.Errorf("Expected %v, but is %v", ts.value, v)
		}
	}
}

func TestSetInputActivations(t *testing.T) {
	network, mb := CreateTestNetwork()
	network.SetInputActivations(LinAlg.MakeVector([]float64{4.9, 3.2}), &mb)

	if a := network.GetActivation(0, &mb); floatEquals(4.9, a.Get(0), EPSILON) == false {
		t.Errorf("Expected 4.9, but is %v", a)
	}

	if a := network.GetActivation(0, &mb); floatEquals(3.2, a.Get(1), EPSILON) == false {
		t.Errorf("Expected 3.2, but is %v", a)
	}
}

func TestCalculateErrorInOutputLayer(t *testing.T) {
	network := CreateTestNetwork2()
	mb := CreateMiniBatch([]int{7, 12})
	network.SetActivation(LinAlg.MakeVector([]float64{1, 2, 3}), 1, &mb)
	outputLayerIdx := network.getOutputLayerIndex()
	costFunction := QuadraticCostFunction{}

	tables := []struct {
		initialOutputActivations  []float64
		expectedOutputActivations []float64
		error                     []float64
	}{
		{[]float64{1, 1}, []float64{0.76, 0.65}, []float64{0.0045536360340586, 0.03509326463138775}},
		{[]float64{0.32, 0.34}, []float64{0.01, 0.78}, []float64{0.005881779877325692, -0.04411724696517317}},
		{[]float64{0.24, 0.21}, []float64{0.66, 0.98}, []float64{-0.007968863059602552, -0.07720518218905305}},
	}

	for _, ts := range tables {
		network.SetActivation(LinAlg.MakeVector(ts.initialOutputActivations), outputLayerIdx, &mb)
		costFunction.CalculateErrorInOutputLayer(&network, LinAlg.MakeVector(ts.expectedOutputActivations), &mb)
		if delta := network.GetDelta(outputLayerIdx, &mb); floatEquals(ts.error[0], delta.Get(0), EPSILON) == false {
			t.Errorf("Expected %v, but was %v", ts.error[0], delta.Get(0))
		}
		if delta := network.GetDelta(outputLayerIdx, &mb); floatEquals(ts.error[1], delta.Get(1), EPSILON) == false {
			t.Errorf("Expected %v, but was %v", ts.error[1], delta.Get(1))
		}
	}
}

func TestBackpropagateError(t *testing.T) {
	network := CreateTestNetwork2()
	mb := CreateMiniBatch([]int{7, 12})
	network.SetActivation(LinAlg.MakeVector([]float64{0.32, 0.56}), 0, &mb)
	costFunction := QuadraticCostFunction{}
	network.Feedforward(&mb)
	costFunction.CalculateErrorInOutputLayer(&network, LinAlg.MakeVector([]float64{1, 0}), &mb)
	network.BackpropagateError(&mb)

	expected := -0.0010048637687567257
	if delta := network.GetDelta(1, &mb); floatEquals(expected, delta.Get(0), EPSILON) == false {
		t.Errorf("Expected %v, but was %v", expected, delta.Get(0))
	}

	expected = 0.018229366486609905
	if delta := network.GetDelta(1, &mb); floatEquals(expected, delta.Get(1), EPSILON) == false {
		t.Errorf("Expected %v, but was %v", expected, delta.Get(1))
	}

	expected = 0.018229366486609905
	if delta := network.GetDelta(1, &mb); floatEquals(0.0010172359440642931, delta.Get(2), EPSILON) == false {
		t.Errorf("Expected %v, but was %v", 0.0010172359440642931, delta.Get(2))
	}
}

func TestCalculateDerivatives(t *testing.T) {
	network, mb := CreateTestNetwork()
	costFunction := QuadraticCostFunction{}
	network.Feedforward(&mb)
	costFunction.CalculateErrorInOutputLayer(&network, LinAlg.MakeVector([]float64{0, 1}), &mb)
	network.BackpropagateError(&mb)

	dw, db := network.CalculateDerivatives([]Minibatch{mb})

	if floatEquals(0, dw[1].Get(0, 0), EPSILON) == false {
		t.Errorf("dw(%v, %v), layer %v, does not equal to %v, but instead %v", 0, 0, 1, 0, dw[1].Get(0, 0))
	}
	if floatEquals(0, db[1].Get(0), EPSILON) == false {
		t.Errorf("db(%v), layer %v, does not equal to %v, but instead %v", 0, 0, 0, db[1].Get(0))
	}
}

func TestTrain(t *testing.T) {
	network, _ := CreateTestNetwork()

	ts := []MNISTImport.TrainingSample{MNISTImport.CreateTrainingSample(LinAlg.MakeVector([]float64{0.34, 0.43}), LinAlg.MakeVector([]float64{0, 1})), MNISTImport.CreateTrainingSample(LinAlg.MakeVector([]float64{0.14, 0.03}), LinAlg.MakeVector([]float64{0, 1}))}
	network.Train(ts, []MNISTImport.TrainingSample{}, 2, 0.001, 0, 10, QuadraticCostFunction{})

	mb := CreateMiniBatch([]int{12, 7})
	network.SetInputActivations(LinAlg.MakeVector([]float64{0.34, 0.43}), &mb)
	network.Feedforward(&mb)

	// Assert
	expected := 0.999999999998501
	if a := network.GetActivation(2, &mb); floatEquals(expected, a.Get(0), EPSILON) == false {
		t.Errorf("Network gave wrong answer. Expected %v, was %v", expected, a.Get(0))
	}

	expected = 0.999999999998501
	if a := network.GetActivation(2, &mb); floatEquals(1, a.Get(1), EPSILON) == false {
		t.Errorf("Network gave wrong answer. Expected %v, was %v", 1, a.Get(1))
	}
}

func TestSingleNeuronQuadraticCostTrain(t *testing.T) {
	network := CreateNetwork([]int{1, 1})
	network.GetWeights(1).Set(0, 0, 2)
	network.GetBias(1).Set(0, 2)
	mb := CreateMiniBatch([]int{2, 1})
	mb.a[0].Set(0, 1)

	ts := []MNISTImport.TrainingSample{MNISTImport.CreateTrainingSample(LinAlg.MakeVector([]float64{1}), LinAlg.MakeVector([]float64{0}))}
	network.Train(ts, []MNISTImport.TrainingSample{}, 300, 0.15, 0, 10, QuadraticCostFunction{})

	network.SetInputActivations(ts[0].InputActivations, &mb)
	network.Feedforward(&mb)

	// Assert
	expected := 0.20284840518811262
	if a := network.GetActivation(1, &mb); floatEquals(expected, a.Get(0), EPSILON) == false {
		t.Errorf("Network gave wrong answer. Expected %v, was %v", expected, a.Get(0))
	}
}

func TestSingleNeuronCrossEntropyCostTrain(t *testing.T) {
	network := CreateNetwork([]int{1, 1})
	network.GetWeights(1).Set(0, 00, 2)
	network.GetBias(1).Set(0, 2)
	mb := CreateMiniBatch([]int{2, 1})
	mb.a[0].Set(0, 1)

	ts := []MNISTImport.TrainingSample{MNISTImport.CreateTrainingSample(LinAlg.MakeVector([]float64{1}), LinAlg.MakeVector([]float64{0}))}
	network.Train(ts, []MNISTImport.TrainingSample{}, 300, 0.5, 0, 10, CrossEntropyCostFunction{})

	network.SetInputActivations(ts[0].InputActivations, &mb)
	network.Feedforward(&mb)

	// Assert
	expected := 0.0033988511270660214
	if a := network.GetActivation(1, &mb); floatEquals(expected, a.Get(0), EPSILON) == false {
		t.Errorf("Network gave wrong answer. Expected %v, was %v", expected, a.Get(0))
	}
}

func TestTrainWithMNIST(t *testing.T) {
	network := CreateNetwork([]int{28 * 28, 100, 10})
	network.InitializeNetworkWeightsAndBiases()

	trainingData := MNISTImport.ImportData("/home/svenschmidt75/Develop/Go/go/src/SimpleNeuralNet/test_data/", "train-images50.idx3-ubyte", "train-labels50.idx1-ubyte")
	ts := trainingData.GenerateTrainingSamples(trainingData.Length())
	network.Train(ts, []MNISTImport.TrainingSample{}, 2, 0.5, 0, 10, QuadraticCostFunction{})

	// Assert
	testData := MNISTImport.ImportData("/home/svenschmidt75/Develop/Go/MNIST", "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
	ts2 := testData.GenerateTrainingSamples(100)

	mb := CreateMiniBatch(network.GetLayers())
	for index := 0; index < len(ts2); index++ {
		network.SetInputActivations(ts2[index].InputActivations, &mb)
		network.Feedforward(&mb)
		as := network.GetOutputLayerActivations(&mb)
		expectedClass := GetClass(ts2[index].OutputActivations)
		fmt.Printf("should %d: %v\n", expectedClass, as)
	}
}

func TestSerialization(t *testing.T) {
	network := CreateNetwork([]int{28 * 28, 100, 10})
	network.InitializeNetworkWeightsAndBiases()

	trainingData := MNISTImport.ImportData("/home/svenschmidt75/Develop/Go/go/src/SimpleNeuralNet/test_data/", "train-images50.idx3-ubyte", "train-labels50.idx1-ubyte")
	ts := trainingData.GenerateTrainingSamples(trainingData.Length())
	network.Train(ts, []MNISTImport.TrainingSample{}, 2, 0.5, 0, 10, QuadraticCostFunction{})

	var buf bytes.Buffer
	err := Utility.WriteGob(&buf, &network)
	if err != nil {
		t.Errorf("Error serializing network")
	}

	readNetwork := new(Network)
	err = Utility.ReadGob(&buf, readNetwork)
	if err != nil {
		t.Error("Error deserializing network")
	}

	a1 := network.GetWeights(1)
	a2 := readNetwork.GetWeights(1)
	if floatEquals(a1.Get(1, 1), a2.Get(1, 1), EPSILON) == false {
		t.Error("Networks not equal")
	}
}
