package main

import (
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

func TestFeedforwardActivation(t *testing.T) {
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

	network.activations[network.GetActivationIndex(0, 0)] = 1
	network.activations[network.GetActivationIndex(1, 0)] = 2

	tables := []struct {
		layer int
		index int
		value float64
	}{
		{1, 0, 1.0},
	}

	for _, ts := range tables {
		v := network.FeedforwardActivation(ts.index, ts.layer)
		if v != ts.value {
			t.Errorf("Expected %v, but is %v", ts.value, v)
		}
	}
}

func TestFeedforward(t *testing.T) {
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

	network.activations[network.GetActivationIndex(0, 0)] = 1
	network.activations[network.GetActivationIndex(1, 0)] = 2

	// initialize weights, biases with random numbers
	network.Feedforward()
}
