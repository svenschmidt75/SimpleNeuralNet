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
		network.SetInputActivations(x.InputActivations, &mb)
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
	l2 := weightsSquared(network)
	l2 *= lambda / float64(2*len(trainingSamples))
	return cost + l2
}

func weightsSquared(n *Network) float64 {
	var l2 float64
	for layer := range n.GetLayers() {
		w := n.GetWeights(layer)
		for row := 0; row < w.Rows; row++ {
			for col := 0; col < w.Cols; col++ {
				v := w.Get(row, col)
				l2 += v
			}
		}
	}
	return l2
}

func caculateDeltaCrossEntropy(layer int, n *Network, mb *Minibatch, ts *MNISTImport.TrainingSample) LinAlg.Vector {
	if layer == n.getOutputLayerIndex() {
		dCda := LinAlg.SubtractVectors(mb.a[layer], ts.OutputActivations)
		return dCda
	}
	weights := n.GetWeights(layer + 1)
	w_transpose := weights.Transpose()
	delta := caculateDeltaCrossEntropy(layer+1, n, mb, ts)
	ax := w_transpose.Ax(delta)
	s := mb.z[layer].F(SigmoidPrime)
	d := ax.Hadamard(&s)
	return d
}

func (CrossEntropyCostFunction) GradBias(layer int, network *Network, trainingSamples []MNISTImport.TrainingSample) LinAlg.Vector {
	// return dC/db for layer l
	if layer == 0 {
		panic(fmt.Sprintf("Layer must be > 0"))
	}
	var delta LinAlg.Vector
	mb := CreateMiniBatch(network.GetLayers())
	for _, x := range trainingSamples {
		network.SetInputActivations(x.InputActivations, &mb)
		network.Feedforward(&mb)
		delta_j := caculateDeltaCrossEntropy(layer, network, &mb, &x)
		delta.Add(delta_j)
	}
	delta.ScalarMultiplication(1 / float64(len(trainingSamples)))
	return delta
}

func (CrossEntropyCostFunction) GradWeight(layer int, lambda float64, network *Network, trainingSamples []MNISTImport.TrainingSample) LinAlg.Matrix {
	// return dC/dw = dC0/dw + lambda / n * w for layer l
	if layer == 0 {
		panic(fmt.Sprintf("Layer must be > 0"))
	}
	var dCdw LinAlg.Matrix
	mb := CreateMiniBatch(network.GetLayers())
	for _, x := range trainingSamples {
		network.SetInputActivations(x.InputActivations, &mb)
		network.Feedforward(&mb)
		a_k := network.GetActivation(layer-1, &mb)
		delta_j := caculateDeltaCrossEntropy(layer, network, &mb, &x)
		tmp := LinAlg.OuterProduct(a_k, delta_j)
		dCdw.Add(tmp)
	}
	dCdw.ScalarMultiplication(1 / float64(len(trainingSamples)))

	// add the regularization term
	w := network.GetWeights(layer)
	w.ScalarMultiplication(lambda / float64(len(trainingSamples)))
	dCdw.Add(*w)

	return dCdw
}

func (CrossEntropyCostFunction) CalculateErrorInOutputLayer(n *Network, outputActivations LinAlg.Vector, mb *Minibatch) {
	// Equation (68), Chapter 3 of http://neuralnetworksanddeeplearning.com
	outputLayerIdx := n.getOutputLayerIndex()
	mb.delta[outputLayerIdx] = LinAlg.SubtractVectors(mb.a[outputLayerIdx], outputActivations)
}
