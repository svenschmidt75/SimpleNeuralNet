package main

import (
	"SimpleNeuralNet/LinAlg"
	"SimpleNeuralNet/MNISTImport"
)

type CostFunction interface {
	Evaluate(network *Network, trainingSamples []MNISTImport.TrainingSample) float64
	GradBias(layer int, network *Network, trainingSamples []MNISTImport.TrainingSample) float64
	GradWeight(layer int, network *Network, trainingSamples []MNISTImport.TrainingSample) float64
	CalculateErrorInOutputLayer(n *Network, outputActivations *LinAlg.Vector, mb *Minibatch)
}
