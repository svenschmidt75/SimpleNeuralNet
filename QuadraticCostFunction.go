package main

import (
	"SimpleNeuralNet/LinAlg"
	"SimpleNeuralNet/MNISTImport"
	"fmt"
)

type QuadraticCostFunction struct{}

// -- Stringer --

func (QuadraticCostFunction) String() string {
	return "Quadratic"
}

// -- CostFunction --

func (QuadraticCostFunction) Evaluate(network *Network, lambda float64, trainingSamples []MNISTImport.TrainingSample) float64 {
	var cost float64
	mb := CreateMiniBatch(network.GetLayers())
	for _, x := range trainingSamples {
		mb.a[0] = x.InputActivations
		network.Feedforward(&mb)
		a := network.GetOutputLayerActivations(&mb)
		diff := GetError(x.OutputActivations, a)
		cost += diff * diff
	}
	fac := float64(2 * len(trainingSamples))
	cost /= fac

	// add the regularization term
	l2 := weightsSquared(network)
	l2 *= lambda / fac
	return cost + l2
}

func calculateDeltaCost(layer int, n *Network, mb *Minibatch, ts *MNISTImport.TrainingSample) *LinAlg.Vector {
	if layer == n.getOutputLayerIndex() {
		delta_L := LinAlg.SubtractVectors(&mb.a[layer], &ts.OutputActivations).Hadamard(mb.z[layer].F(SigmoidPrime))
		return delta_L
	}
	delta_next := calculateDeltaCost(layer+1, n, mb, ts)
	s := mb.z[layer].F(SigmoidPrime)
	delta := n.GetWeights(layer + 1).Transpose().Ax(delta_next).Hadamard(s)
	return delta
}

func (QuadraticCostFunction) GradBias(layer int, network *Network, trainingSamples []MNISTImport.TrainingSample) *LinAlg.Vector {
	if layer == 0 {
		panic(fmt.Sprintf("Layer must be > 0"))
	}
	var delta LinAlg.Vector
	mb := CreateMiniBatch(network.GetLayers())
	for _, x := range trainingSamples {
		mb.a[0] = x.InputActivations
		network.Feedforward(&mb)
		delta_j := calculateDeltaCost(layer, network, &mb, &x)
		delta.Add(delta_j)
	}
	delta.Scalar(1 / float64(len(trainingSamples)))
	return &delta
}

func (QuadraticCostFunction) GradWeight(layer int, lambda float64, network *Network, trainingSamples []MNISTImport.TrainingSample) *LinAlg.Matrix {
	if layer == 0 {
		panic(fmt.Sprintf("Layer must be > 0"))
	}
	var dCdw LinAlg.Matrix
	mb := CreateMiniBatch(network.GetLayers())
	for _, x := range trainingSamples {
		mb.a[0] = x.InputActivations
		network.Feedforward(&mb)
		delta_j := calculateDeltaCost(layer, network, &mb, &x)
		tmp := LinAlg.OuterProduct(&mb.a[layer-1], delta_j)
		dCdw.Add(tmp)
	}
	dCdw.Scalar(1 / float64(len(trainingSamples)))

	// add the regularization term
	l2 := network.GetWeights(layer).Scalar(lambda / float64(len(trainingSamples)))
	dCdw.Add(l2)

	return &dCdw
}

func (QuadraticCostFunction) CalculateErrorInOutputLayer(n *Network, outputActivations *LinAlg.Vector, mb *Minibatch) {
	// Equation (BP1) and (30), Chapter 2 of http://neuralnetworksanddeeplearning.com
	outputLayerIdx := n.getOutputLayerIndex()
	delta := LinAlg.SubtractVectors(&mb.a[outputLayerIdx], outputActivations).Hadamard(mb.z[outputLayerIdx].F(SigmoidPrime))
	mb.delta[outputLayerIdx] = *delta
}
