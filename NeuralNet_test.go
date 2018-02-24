package main

import (
	//"math"
	"testing"
	"math"
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
		network := CreateNetwork(item.xs, 1)
		nWeights := len(network.batches[0].w)
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
		network := CreateNetwork(item.xs, 1)
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
		network := CreateNetwork(ts.xs, 1)
		for _, as := range ts.ais {
			ai := network.GetActivationIndex(as.i, as.layer)
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
		network := CreateNetwork(ts.xs, 1)
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
		network := CreateNetwork(ts.xs, 1)
		for _, as := range ts.ais {
			ai := network.GetWeightIndex(as.i, as.j, as.layer)
			if ai != as.index {
				t.Errorf("Expected %v, but is %v", as.index, ai)
			}
		}
	}
}

func CreateTestNetwork(nMiniBatches int) Network {
	network := CreateNetwork([]int{2, 3, 2}, nMiniBatches)

	network.batches[0].w[network.GetWeightIndex(0, 0, 1)] = 1
	network.batches[0].w[network.GetWeightIndex(0, 1, 1)] = 2
	network.batches[0].w[network.GetWeightIndex(1, 0, 1)] = 3
	network.batches[0].w[network.GetWeightIndex(1, 1, 1)] = 4
	network.batches[0].w[network.GetWeightIndex(2, 0, 1)] = 5
	network.batches[0].w[network.GetWeightIndex(2, 1, 1)] = 6
	network.batches[0].w[network.GetWeightIndex(0, 0, 2)] = 7
	network.batches[0].w[network.GetWeightIndex(0, 1, 2)] = 8
	network.batches[0].w[network.GetWeightIndex(0, 2, 2)] = 9
	network.batches[0].w[network.GetWeightIndex(1, 0, 2)] = 10
	network.batches[0].w[network.GetWeightIndex(1, 1, 2)] = 11
	network.batches[0].w[network.GetWeightIndex(1, 2, 2)] = 12

	network.biases[network.GetBiasIndex(0, 1)] = 1
	network.biases[network.GetBiasIndex(1, 1)] = 2
	network.biases[network.GetBiasIndex(2, 1)] = 3
	network.biases[network.GetBiasIndex(0, 2)] = 4
	network.biases[network.GetBiasIndex(1, 2)] = 5

	network.batches[0].a[network.GetActivationIndex(0, 0)] = 1
	network.batches[0].a[network.GetActivationIndex(1, 0)] = 2

	return network
}

func TestCalculateZ(t *testing.T) {
	network := CreateTestNetwork(1)

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
		v := network.CalculateZ(ts.index, ts.layer, 0)

		// for this test, we assume the activation function is the identity
		*network.GetActivation(ts.index, ts.layer, 0) = v
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
	network := CreateTestNetwork(1)

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
		v := network.FeedforwardActivation(ts.index, ts.layer, 0)
		*network.GetActivation(ts.index, ts.layer, 0) = v
		if floatEquals(v, ts.value) == false {
			t.Errorf("Expected %v, but is %v", ts.value, v)
		}
	}
}

func TestFeedforward(t *testing.T) {
	network := CreateTestNetwork(1)
	network.Feedforward(0)

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
		v := *network.GetActivation(ts.index, ts.layer, 0)
		if floatEquals(v, ts.value) == false {
			t.Errorf("Expected %v, but is %v", ts.value, v)
		}
	}
}

func TestSetInputActivations(t *testing.T) {
	network := CreateTestNetwork(2)

	network.SetInputActivations([]float64{4.9, 3.2}, 1)

	if a := *network.GetActivation(0, 0, 1); floatEquals(4.9, a) == false {
		t.Errorf("Expected 4.9, but is %v", a)
	}

	if a := *network.GetActivation(1, 0, 1); floatEquals(3.2, a) == false {
		t.Errorf("Expected 3.2, but is %v", a)
	}
}

func TestCalculateErrorInOutputLayer(t *testing.T) {
	network := CreateTestNetwork(1)
	network.Feedforward(0)

	errorInOutputLayer := network.CalculateErrorInOutputLayer([]float64{0.1, 0.5}, 0)

	if l := len(errorInOutputLayer); l != 2 {
		t.Errorf("Number of error elements %v not equal to 2", l)
	}

	if floatEquals(errorInOutputLayer[0], 1E-13) == false {
		t.Errorf("Expected %v, but is %v", errorInOutputLayer[0], 1E-13)
	}
}

func TestBackpropagate(t *testing.T) {
	network := CreateTestNetwork(1)
	network.Feedforward(0)
	nabla_L := network.CalculateErrorInOutputLayer([]float64{0.1, 0.5}, 0)

	nablas := network.Backpropagate(nabla_L)

	if l := len(nablas); l != 2 {
		t.Errorf("Number of error elements %v not equal to 12", l)
	}
	if l := len(nablas[0]); l != 3 {
		t.Errorf("Number of error elements %v not equal to 2", l)
	}
	if l := len(nablas[1]); l != 2 {
		t.Errorf("Number of error elements %v not equal to 3", l)
	}
}

func TestUpdateNetwork(t *testing.T) {

}
