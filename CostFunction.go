package main

import (
	"SimpleNeuralNet/LinAlg"
	"SimpleNeuralNet/MNISTImport"
)

type CostFunction interface {
	Evaluate(network *Network, lambda float64, trainingSamples []MNISTImport.TrainingSample) float64
	GradBias(layer int, network *Network, trainingSamples []MNISTImport.TrainingSample) *LinAlg.Vector
	GradWeight(layer int, lambda float64, network *Network, trainingSamples []MNISTImport.TrainingSample) *LinAlg.Matrix
	CalculateErrorInOutputLayer(n *Network, outputActivations *LinAlg.Vector, mb *Minibatch)
}
