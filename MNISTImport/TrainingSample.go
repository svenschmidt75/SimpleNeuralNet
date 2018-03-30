package MNISTImport

type TrainingSample struct {
	// Input layer activations
	InputActivations []float64

	// Expected class
	ExpectedClass int
}

func CreateTrainingSample(inputActivations []float64, expectedClass int) TrainingSample {
	ts := TrainingSample{inputActivations, expectedClass}
	return ts
}
