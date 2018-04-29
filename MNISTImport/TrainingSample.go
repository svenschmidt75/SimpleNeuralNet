package MNISTImport

import "SimpleNeuralNet/LinAlg"

type TrainingSample struct {
	// Input layer activations
	InputActivations LinAlg.Vector

	// Expected output
	OutputActivations LinAlg.Vector
}

func CreateTrainingSample(inputActivations *LinAlg.Vector, outputActivations *LinAlg.Vector) TrainingSample {
	ts := TrainingSample{*inputActivations, *outputActivations}
	return ts
}
