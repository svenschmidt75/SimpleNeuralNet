package main

type TrainingSample struct {
	// Input layer activations
	inputActivations []float64

	// Expected output
	outputActivations []float64
}

func CreateTrainingSample(inputActivations []float64, outputActivations []float64) TrainingSample {
	ts := TrainingSample{inputActivations, outputActivations}
	return ts
}
