package main

type Minibatch struct {
	z []float64

	// activations
	a []float64

	// weights
	w []float64

	// errors
	nabla []float64
}

func CreateMiniBatch(nActivations int, nWeights int) Minibatch {
	w := make([]float64, nWeights)
	a := make([]float64, nActivations)
	z := make([]float64, nActivations)
	nabla := make([]float64, nActivations)
	return Minibatch{z, a, w, nabla}
}