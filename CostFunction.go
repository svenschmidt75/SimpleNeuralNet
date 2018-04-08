package main

import "SimpleNeuralNet/MNISTImport"

type CostFunction interface {
	Evaluate(network *Network, trainingSamples []MNISTImport.TrainingSample) float64
	GradBias(j int, layer int, network *Network, trainingSamples []MNISTImport.TrainingSample) float64
	GradWeight(j int, k int, layer int, network *Network, trainingSamples []MNISTImport.TrainingSample) float64
	CalculateErrorInOutputLayer(n *Network, expectedClass int, mb *Minibatch)
}
