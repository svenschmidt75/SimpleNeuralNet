package main

import (
	"SimpleNeuralNet/LinAlg"
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
		w_jk := network.GetWeights(item.layer)
		value := w_jk.Get(item.i, item.j)
		w_jk.Set(item.i, item.j, value-delta)
		c1 := costFunction.Evaluate(network, lambda, ts)
		w_jk.Set(item.i, item.j, value+delta)
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
		b := network.GetBias(item.layer)
		value := b.Get(item.i)
		b.Set(item.i, value-delta)
		c1 := costFunction.Evaluate(network, lambda, ts)
		b.Set(item.i, value+delta)
		c2 := costFunction.Evaluate(network, lambda, ts)
		dCdb_numeric := (c2 - c1) / 2 / delta

		// evaluate analytically
		dCdb := costFunction.GradBias(item.layer, network, ts)

		if floatEquals(dCdb_numeric, dCdb.Get(item.i), EPSILON) == false {
			t.Error("Networks not equal")
		}
	}
}

func TestCrossEntropyErrorOutputLayerNumerically(t *testing.T) {
	network := CreateNetwork([]int{1, 1})
	network.GetWeights(1).Set(0, 0, 2)
	network.GetBias(1).Set(0, 2)
	mb := CreateMiniBatch([]int{2})
	mb.a[0].Set(0, 1)
	costFunction := CrossEntropyCostFunction{}
	lambda := float64(1)

	y := LinAlg.MakeVector([]float64{0})
	ts := []MNISTImport.TrainingSample{MNISTImport.CreateTrainingSample(LinAlg.MakeVector([]float64{1}), y)}
	network.Train(ts, []MNISTImport.TrainingSample{}, 300, 0.15, lambda, 10, costFunction)
	mb.a[0] = ts[0].InputActivations
	network.Feedforward(&mb)
	costFunction.CalculateErrorInOutputLayer(&network, &ts[0].OutputActivations, &mb)

	C := func(z *LinAlg.Vector, y *LinAlg.Vector) float64 {
		a := z.F(Sigmoid)
		var cost float64
		for j := 0; j < a.Size(); j++ {
			yj := y.Get(j)
			aj := a.Get(j)
			var term float64
			term = yj*math.Log(aj) + (1-yj)*math.Log(1-aj)
			cost += term
		}
		return cost
	}
	delta := 0.000001
	z_j := mb.z[0]
	z_j.Set(0, z_j.Get(0)-delta)
	c1 := C(&z_j, y)
	z_j.Set(0, z_j.Get(0)+delta)
	c2 := C(&z_j, y)
	dCdb_numeric := (c2 - c1) / 2 / delta

	// evaluate analytically
	delta_L := mb.delta[0]

	if floatEquals(dCdb_numeric, delta_L.Get(0), EPSILON) == false {
		t.Error("Networks not equal")
	}
}
