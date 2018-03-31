package main

type Minibatch struct {
	z []float64

	// activations
	a []float64

	// errors
	delta []float64
}

func CreateMiniBatch(nActivations int, nWeights int) Minibatch {
	a := make([]float64, nActivations)
	z := make([]float64, nActivations)
	nabla := make([]float64, nActivations)
	return Minibatch{z, a, nabla}
}

func CreateMiniBatches(size int, nActivations int, nWeights int) []Minibatch {
	mbs := make([]Minibatch, size)
	for idx := range mbs {
		mbs[idx] = CreateMiniBatch(nActivations, nWeights)
	}
	return mbs
}
