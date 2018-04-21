package main

import (
	"SimpleNeuralNet/MNISTImport"
	"testing"
)

func TestQuadraticCostDerivativeWeightNumerical(t *testing.T) {
	// Arrange
	network := new(Network)
	err := ReadGobFromFile("./50000_30_3_10.gob", network)
	if err != nil {
		t.Error("Error deserializing network")
	}
	network.Lambda = 1
	costFunction := QuadtraticCostFunction{}
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
		c1 := costFunction.Evaluate(network, ts)
		network.SetWeight(w_jk+delta, item.i, item.j, item.layer)
		c2 := costFunction.Evaluate(network, ts)
		dCdw_numeric := (c2 - c1) / 2 / delta

		// evaluate analytically
		dCdw := costFunction.GradWeight(item.i, item.j, item.layer, network, ts)

		if floatEquals(dCdw_numeric, dCdw, EPSILON*10) == false {
			t.Error("Networks not equal")
		}
	}
}

func TestQuadraticCostDerivativeBiasNumerical(t *testing.T) {
	// Arrange
	network := new(Network)
	err := ReadGobFromFile("./50000_30_3_10.gob", network)
	if err != nil {
		t.Error("Error deserializing network")
	}
	costFunction := QuadtraticCostFunction{}
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
		c1 := costFunction.Evaluate(network, ts)
		network.SetBias(b_j+delta, item.i, item.layer)
		c2 := costFunction.Evaluate(network, ts)
		dCdb_numeric := (c2 - c1) / 2 / delta

		// evaluate analytically
		dCdb := costFunction.GradBias(item.i, item.layer, network, ts)

		if floatEquals(dCdb_numeric, dCdb, EPSILON) == false {
			t.Error("Networks not equal")
		}
	}
}

func TestQuadraticCostErrorOutputLayerNumerically(t *testing.T) {
	network := CreateNetwork([]int{1, 1}, 0)
	network.weights[network.GetWeightIndex(0, 0, 1)] = 2
	network.biases[network.GetBiasIndex(0, 1)] = 2
	mb := CreateMiniBatch(2, 1)
	mb.a[network.GetNodeIndex(0, 0)] = 1
	costFunction := QuadtraticCostFunction{}

	ts := []MNISTImport.TrainingSample{MNISTImport.CreateTrainingSample([]float64{1}, []float64{0})}
	network.Train(ts, []MNISTImport.TrainingSample{}, 300, 0.15, 10, costFunction)
	network.SetInputActivations(ts[0].InputActivations, &mb)
	network.Feedforward(&mb)
	costFunction.CalculateErrorInOutputLayer(&network, ts[0].OutputActivations, &mb)

	C := func(z float64) float64 {
		a := Sigmoid(z)
		return 0.5 * a * a
	}
	delta := 0.000001
	z_j := network.CalculateZ(0, 1, &mb)
	c1 := C(z_j - delta)
	c2 := C(z_j + delta)
	dCdb_numeric := (c2 - c1) / 2 / delta

	// evaluate analytically
	delta_L := network.GetDelta(0, 1, &mb)

	if floatEquals(dCdb_numeric, delta_L, EPSILON) == false {
		t.Error("Networks not equal")
	}
}
