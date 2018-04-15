package MNISTImport

type TrainingSample struct {
	// Input layer activations
	InputActivations []float64

	// Expected output
	OutputActivations []float64
}

func CreateTrainingSample(inputActivations []float64, outputActivations []float64) TrainingSample {
	ts := TrainingSample{inputActivations, outputActivations}
	return ts
}
