package main

import "SimpleNeuralNet/LinAlg"

type Minibatch struct {
	z []LinAlg.Vector

	// activations
	a []LinAlg.Vector

	// errors
	delta []LinAlg.Vector
}

func CreateMiniBatch(layers []int) Minibatch {
	z := createVectors(layers)
	a := createVectors(layers)
	delta := createVectors(layers)
	return Minibatch{z, a, delta}
}

func CreateMiniBatches(size int, layers []int) []Minibatch {
	mbs := make([]Minibatch, size)
	for idx := range mbs {
		mbs[idx] = CreateMiniBatch(layers)
	}
	return mbs
}

func createVectors(layers []int) []LinAlg.Vector {
	result := make([]LinAlg.Vector, len(layers))
	for idx, nNodes := range layers[1:] {
		result[idx] = LinAlg.MakeEmptyVector(nNodes)
	}
	return result
}
