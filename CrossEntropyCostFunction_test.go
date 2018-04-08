package main

import (
	"SimpleNeuralNet/MNISTImport"
	"testing"
)

func TestCrossEntropyCostDerivativeWeightNumerical(t *testing.T) {
	// Arrange
	network := new(Network)
	err := ReadGobFromFile("./50000_30_3_10.gob", network)
	if err != nil {
		t.Error("Error deserializing network")
	}
	trainingData := MNISTImport.ImportData("./test_data/", "train-images50.idx3-ubyte", "train-labels50.idx1-ubyte")
	ts := trainingData.GenerateTrainingSamples(trainingData.Length())

	// Act
	tables := []struct {
		i     int
		j     int
		layer int
	}{
		{1, 17, 2},
		{3, 22, 2},
		{5, 56, 2},
		{7, 78, 2},
		{9, 98, 2},
		{98, 498, 1},
		{38, 780, 1},
		{18, 281, 1},
		{4, 81, 1},
	}
	for _, item := range tables {
		// evaluate numerically
		delta := 0.000001
		w_jk := network.GetWeight(item.i, item.j, item.layer)
		network.SetWeight(w_jk-delta, item.i, item.j, item.layer)
		c1 := network.CostFunction.Evaluate(network, ts)
		network.SetWeight(w_jk+delta, item.i, item.j, item.layer)
		c2 := network.CostFunction.Evaluate(network, ts)
		dCdw_numeric := (c2 - c1) / 2 / delta

		// evaluate analytically
		dCdw := network.CostFunction.GradWeight(item.i, item.j, item.layer, network, ts)

		if floatEquals(dCdw_numeric, dCdw) == false {
			t.Error("Networks not equal")
		}
	}
}

func TestCrossEntropyCostDerivativeBiasNumerical(t *testing.T) {
	// Arrange
	network := new(Network)
	err := ReadGobFromFile("./50000_30_3_10.gob", network)
	if err != nil {
		t.Error("Error deserializing network")
	}
	trainingData := MNISTImport.ImportData("./test_data/", "train-images50.idx3-ubyte", "train-labels50.idx1-ubyte")
	ts := trainingData.GenerateTrainingSamples(trainingData.Length())

	// Act
	tables := []struct {
		i     int
		j     int
		layer int
	}{
		{1, 17, 2},
		{3, 22, 2},
		{5, 56, 2},
		{7, 78, 2},
		{9, 98, 2},
		{98, 498, 1},
		{38, 780, 1},
		{18, 281, 1},
		{4, 81, 1},
	}
	for _, item := range tables {
		// evaluate numerically
		delta := 0.000001
		b_j := network.GetBias(item.i, item.layer)
		network.SetBias(b_j-delta, item.i, item.layer)
		c1 := network.CostFunction.Evaluate(network, ts)
		network.SetBias(b_j+delta, item.i, item.layer)
		c2 := network.CostFunction.Evaluate(network, ts)
		dCdb_numeric := (c2 - c1) / 2 / delta

		// evaluate analytically
		dCdb := network.CostFunction.GradBias(item.i, item.layer, network, ts)

		if floatEquals(dCdb_numeric, dCdb) == false {
			t.Error("Networks not equal")
		}
	}
}
