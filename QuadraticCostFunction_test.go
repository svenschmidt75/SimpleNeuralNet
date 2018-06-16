package main

import (
	"SimpleNeuralNet/LinAlg"
	"SimpleNeuralNet/MNISTImport"
	"SimpleNeuralNet/Utility"
	"testing"
)

func TestQuadraticCostDerivativeWeightNumerical(t *testing.T) {
	// Arrange
	network := new(Network)
	err := Utility.ReadGobFromFile("./54000_30_3_25 - 28^2 x 200 x 10_CE.gob", network)
	if err != nil {
		t.Fatal("Error deserializing network")
	}
	costFunction := QuadraticCostFunction{}
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

func TestQuadraticCostDerivativeBiasNumerical(t *testing.T) {
	// Arrange
	network := new(Network)
	err := Utility.ReadGobFromFile("./50000_30_3_10.gob", network)
	if err != nil {
		t.Error("Error deserializing network")
	}
	costFunction := QuadraticCostFunction{}
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

func TestQuadraticCostErrorOutputLayerNumerically(t *testing.T) {
	network := CreateNetwork([]int{1, 1})
	network.GetWeights(1).Set(0, 0, 2)
	network.GetBias(1).Set(0, 2)
	mb := CreateMiniBatch([]int{1, 1})
	mb.a[0].Set(0, 1)
	costFunction := QuadraticCostFunction{}
	lambda := float64(1)

	ts := []MNISTImport.TrainingSample{MNISTImport.CreateTrainingSample(LinAlg.MakeVector([]float64{1}), LinAlg.MakeVector([]float64{0}))}
	network.Train(ts, []MNISTImport.TrainingSample{}, 300, 0.15, lambda, 10, costFunction)
	mb.a[0] = ts[0].InputActivations
	network.Feedforward(&mb)
	costFunction.CalculateErrorInOutputLayer(&network, &ts[0].OutputActivations, &mb)

	C := func(z *LinAlg.Vector) float64 {
		a := z.F(Sigmoid)
		return 0.5 * a.DotProduct(a)
	}
	delta := 0.000001
	network.CalculateZ(1, &mb)
	z_j := mb.z[1]
	z_j.Set(0, z_j.Get(0)-delta)
	c1 := C(&z_j)
	z_j.Set(0, z_j.Get(0)+2*delta)
	c2 := C(&z_j)
	dCdb_numeric := (c2 - c1) / 2 / delta

	// evaluate analytically
	delta_L := mb.delta[1]

	if floatEquals(dCdb_numeric, delta_L.Get(0), EPSILON) == false {
		t.Error("Networks not equal")
	}
}
