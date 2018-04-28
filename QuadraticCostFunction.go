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
		network.SetInputActivations(x.InputActivations, &mb)
		network.Feedforward(&mb)
		a := network.GetOutputLayerActivations(&mb)
		diff := GetError(&x.OutputActivations, &a)
		cost += diff * diff
	}
	fac := float64(2 * len(trainingSamples))
	cost /= fac

	// add the regularization term
	l2 := weightsSquared(network)
	l2 *= lambda / fac
	return cost + l2
}

func caculateDeltaCost(layer int, n *Network, mb *Minibatch, ts *MNISTImport.TrainingSample) LinAlg.Vector {
	if layer == n.getOutputLayerIndex() {
		dCda := LinAlg.SubtractVectors(&mb.a[layer], &ts.OutputActivations)
		s := mb.z[layer].F(SigmoidPrime)
		d := dCda.Hadamard(&s)
		return d
	}
	w_transpose := n.GetWeights(layer + 1).Transpose()
	delta := caculateDeltaCost(layer+1, n, mb, ts)
	s := mb.z[layer].F(SigmoidPrime)
	ax := w_transpose.Ax(&delta)
	d := ax.Hadamard(&s)
	return d
}

func (QuadraticCostFunction) GradBias(layer int, network *Network, trainingSamples []MNISTImport.TrainingSample) LinAlg.Vector {
	if layer == 0 {
		panic(fmt.Sprintf("Layer must be > 0"))
	}
	var delta LinAlg.Vector
	mb := CreateMiniBatch(network.GetLayers())
	for _, x := range trainingSamples {
		network.SetInputActivations(x.InputActivations, &mb)
		network.Feedforward(&mb)
		delta_j := caculateDeltaCost(layer, network, &mb, &x)
		delta.Add(&delta_j)
	}
	delta.ScalarMultiplication(1 / float64(len(trainingSamples)))
	return delta
}

func (QuadraticCostFunction) GradWeight(layer int, lambda float64, network *Network, trainingSamples []MNISTImport.TrainingSample) LinAlg.Matrix {
	if layer == 0 {
		panic(fmt.Sprintf("Layer must be > 0"))
	}
	var dCdw LinAlg.Matrix
	mb := CreateMiniBatch(network.GetLayers())
	for _, x := range trainingSamples {
		network.SetInputActivations(x.InputActivations, &mb)
		network.Feedforward(&mb)
		a_k := network.GetActivation(layer-1, &mb)
		delta_j := caculateDeltaCost(layer, network, &mb, &x)
		tmp := LinAlg.OuterProduct(a_k, &delta_j)
		dCdw.Add(&tmp)
	}
	dCdw.ScalarMultiplication(1 / float64(len(trainingSamples)))

	// add the regularization term
	w := network.GetWeights(layer)
	w.ScalarMultiplication(lambda / float64(len(trainingSamples)))
	dCdw.Add(w)

	return dCdw
}

func (QuadraticCostFunction) CalculateErrorInOutputLayer(n *Network, outputActivations *LinAlg.Vector, mb *Minibatch) {
	// Equation (BP1) and (30), Chapter 2 of http://neuralnetworksanddeeplearning.com
	outputLayerIdx := n.getOutputLayerIndex()
	delta := LinAlg.SubtractVectors(&mb.a[outputLayerIdx], outputActivations)
	s := mb.z[outputLayerIdx].F(SigmoidPrime)
	d := delta.Hadamard(&s)
	mb.delta[outputLayerIdx] = d
}
