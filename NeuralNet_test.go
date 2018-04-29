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
		mb.a[ts.layer] = v
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
		a := mb.a[ts.layer]
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
		a := mb.a[ts.layer]
		if floatEquals(a.Get(ts.index), ts.value, EPSILON) == false {
			t.Errorf("Expected %v, but is %v", ts.value, a.Get(ts.index))
		}
	}
}

func TestCalculateErrorInOutputLayer(t *testing.T) {
	network := CreateTestNetwork2()
	mb := CreateMiniBatch([]int{7, 12})
	mb.a[1] = *LinAlg.MakeVector([]float64{1, 2, 3})
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
		mb.a[outputLayerIdx] = *LinAlg.MakeVector(ts.initialOutputActivations)
		costFunction.CalculateErrorInOutputLayer(&network, LinAlg.MakeVector(ts.expectedOutputActivations), &mb)
		if delta := mb.delta[outputLayerIdx]; floatEquals(ts.error[0], delta.Get(0), EPSILON) == false {
			t.Errorf("Expected %v, but was %v", ts.error[0], delta.Get(0))
		}
		if delta := mb.delta[outputLayerIdx]; floatEquals(ts.error[1], delta.Get(1), EPSILON) == false {
			t.Errorf("Expected %v, but was %v", ts.error[1], delta.Get(1))
		}
	}
}

func TestBackpropagateError(t *testing.T) {
	network := CreateTestNetwork2()
	mb := CreateMiniBatch([]int{7, 12})
	mb.a[0] = *LinAlg.MakeVector([]float64{0.32, 0.56})
	costFunction := QuadraticCostFunction{}
	network.Feedforward(&mb)
	costFunction.CalculateErrorInOutputLayer(&network, LinAlg.MakeVector([]float64{1, 0}), &mb)
	network.BackpropagateError(&mb)

	expected := -0.0010048637687567257
	if delta := mb.delta[1]; floatEquals(expected, delta.Get(0), EPSILON) == false {
		t.Errorf("Expected %v, but was %v", expected, delta.Get(0))
	}

	expected = 0.018229366486609905
	if delta := mb.delta[1]; floatEquals(expected, delta.Get(1), EPSILON) == false {
		t.Errorf("Expected %v, but was %v", expected, delta.Get(1))
	}

	expected = 0.018229366486609905
	if delta := mb.delta[1]; floatEquals(0.0010172359440642931, delta.Get(2), EPSILON) == false {
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
	mb.a[0] = *LinAlg.MakeVector([]float64{0.34, 0.43})
	network.Feedforward(&mb)

	// Assert
	expected := 0.999999999998501
	if a := mb.a[2]; floatEquals(expected, a.Get(0), EPSILON) == false {
		t.Errorf("Network gave wrong answer. Expected %v, was %v", expected, a.Get(0))
	}

	expected = 0.999999999998501
	if a := mb.a[2]; floatEquals(1, a.Get(1), EPSILON) == false {
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

	mb.a[0] = ts[0].InputActivations
	network.Feedforward(&mb)

	// Assert
	expected := 0.20284840518811262
	if a := mb.a[1]; floatEquals(expected, a.Get(0), EPSILON) == false {
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

	mb.a[0] = ts[0].InputActivations
	network.Feedforward(&mb)

	// Assert
	expected := 0.0033988511270660214
	if a := mb.a[1]; floatEquals(expected, a.Get(0), EPSILON) == false {
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
		mb.a[0] = ts2[index].InputActivations
		network.Feedforward(&mb)
		as := network.GetOutputLayerActivations(&mb)
		expectedClass := GetClass(&ts2[index].OutputActivations)
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
