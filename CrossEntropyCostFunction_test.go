package main

import (
	"SimpleNeuralNet/MNISTImport"
	"SimpleNeuralNet/Utility"
	"math"
	"testing"
)

func TestCrossEntropyCostDerivativeWeightNumerical(t *testing.T) {
	// Arrange
	network := new(Network)
	err := Utility.ReadGobFromFile("./50000_30_3_10.gob", network)
	if err != nil {
		t.Error("Error deserializing network")
	}
	costFunction := CrossEntropyCostFunction{}
	var lambda float64
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
		c1 := costFunction.Evaluate(network, lambda, ts)
		network.SetWeight(w_jk+delta, item.i, item.j, item.layer)
		c2 := costFunction.Evaluate(network, lambda, ts)
		dCdw_numeric := (c2 - c1) / 2 / delta

		// evaluate analytically
		dCdw := costFunction.GradWeight(item.layer, lambda, network, ts)

		if floatEquals(dCdw_numeric, dCdw.Get(item.i, item.j), EPSILON*10) == false {
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
	costFunction := CrossEntropyCostFunction{}
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

func TestCrossEntropyErrorOutputLayerNumerically(t *testing.T) {
	network := CreateNetwork([]int{1, 1}, 1)
	network.weights[network.GetWeightIndex(0, 0, 1)] = 2
	network.biases[network.GetBiasIndex(0, 1)] = 2
	mb := CreateMiniBatch(2, 1)
	mb.a[network.GetNodeIndex(0, 0)] = 1
	costFunction := CrossEntropyCostFunction{}

	ts := []MNISTImport.TrainingSample{MNISTImport.CreateTrainingSample([]float64{1}, []float64{0})}
	network.Train(ts, []MNISTImport.TrainingSample{}, 300, 0.15, 10, costFunction)
	network.SetInputActivations(ts[0].InputActivations, &mb)
	network.Feedforward(&mb)
	costFunction.CalculateErrorInOutputLayer(&network, ts[0].OutputActivations, &mb)

	C := func(z float64, y float64) float64 {
		a := Sigmoid(z)
		return -(y*math.Log(a) + (1-y)*math.Log(1-a))
	}
	delta := 0.000001
	z_j := network.CalculateZ(0, 1, &mb)
	c1 := C(z_j-delta, 0)
	c2 := C(z_j+delta, 0)
	dCdb_numeric := (c2 - c1) / 2 / delta

	// evaluate analytically
	delta_L := network.GetDelta(0, 1, &mb)

	if floatEquals(dCdb_numeric, delta_L, EPSILON) == false {
		t.Error("Networks not equal")
	}
}
