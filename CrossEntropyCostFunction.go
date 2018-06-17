package main

import (
	"SimpleNeuralNet/LinAlg"
	"SimpleNeuralNet/MNISTImport"
	"fmt"
	"math"
)

type CrossEntropyCostFunction struct{}

// -- Stringer --

func (CrossEntropyCostFunction) String() string {
	return "Cross-Entropy"
}

// -- CostFunction --

func (CrossEntropyCostFunction) Evaluate(network *Network, lambda float64, trainingSamples []MNISTImport.TrainingSample) float64 {
	var cost float64
	mb := CreateMiniBatch(network.nodes)
	for _, x := range trainingSamples {
		mb.a[0] = x.InputActivations
		network.Feedforward(&mb)
		a := network.GetOutputLayerActivations(&mb)
		y := x.OutputActivations
		var sumj float64
		for j := 0; j < a.Size(); j++ {
			yj := y.Get(j)
			aj := a.Get(j)
			var term float64
			term = yj*math.Log(aj) + (1-yj)*math.Log(1-aj)
			sumj += term
		}
		cost += sumj
	}
	cost /= -float64(len(trainingSamples))

	// add the regularization term
	l2 := network.weightsSquared()
	l2 *= lambda / float64(2*len(trainingSamples))
	return cost + l2
}

func calculateDeltaCrossEntropy(layer int, n *Network, mb *Minibatch, ts *MNISTImport.TrainingSample) *LinAlg.Vector {
	if layer == n.getOutputLayerIndex() {
		dCda := LinAlg.SubtractVectors(&mb.a[layer], &ts.OutputActivations)
		return dCda
	}
	delta_next := calculateDeltaCrossEntropy(layer+1, n, mb, ts)
	s := mb.z[layer].F(SigmoidPrime)
	delta := n.GetWeights(layer + 1).Transpose().Ax(delta_next).Hadamard(s)
	return delta
}

func (CrossEntropyCostFunction) GradBias(layer int, network *Network, trainingSamples []MNISTImport.TrainingSample) *LinAlg.Vector {
	// return dC/db for layer l
	if layer == 0 {
		panic(fmt.Sprintf("Layer must be > 0"))
	}
	var delta LinAlg.Vector
	mb := CreateMiniBatch(network.GetLayers())
	for _, x := range trainingSamples {
		mb.a[0] = x.InputActivations
		network.Feedforward(&mb)
		delta_j := calculateDeltaCrossEntropy(layer, network, &mb, &x)
		delta.Add(delta_j)
	}
	delta.Scalar(1 / float64(len(trainingSamples)))
	return &delta
}

func (CrossEntropyCostFunction) GradWeight(layer int, lambda float64, network *Network, trainingSamples []MNISTImport.TrainingSample) *LinAlg.Matrix {
	// return dC/dw = dC0/dw + lambda / n * w for layer l
	if layer == 0 {
		panic(fmt.Sprintf("Layer must be > 0"))
	}
	var dCdw LinAlg.Matrix
	mb := CreateMiniBatch(network.GetLayers())
	for _, x := range trainingSamples {
		mb.a[0] = x.InputActivations
		network.Feedforward(&mb)
		delta_j := calculateDeltaCrossEntropy(layer, network, &mb, &x)
		tmp := LinAlg.OuterProduct(&mb.a[layer-1], delta_j)
		dCdw.Add(tmp)
	}
	dCdw.Scalar(1 / float64(len(trainingSamples)))

	// add the regularization term
	l2 := network.GetWeights(layer).Scalar(lambda / float64(len(trainingSamples)))
	dCdw.Add(l2)

	return &dCdw
}

func (CrossEntropyCostFunction) CalculateErrorInOutputLayer(n *Network, outputActivations *LinAlg.Vector, mb *Minibatch) {
	// Equation (68), Chapter 3 of http://neuralnetworksanddeeplearning.com
	outputLayerIdx := n.getOutputLayerIndex()
	mb.delta[outputLayerIdx] = *LinAlg.SubtractVectors(&mb.a[outputLayerIdx], outputActivations)
}
