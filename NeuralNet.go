package main

import (
	"math"
)

type Network struct {
	nLayers int
	layers  []int
	biases  []float64
	weights []float64
}

func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

func (n Network) Feedforward() []float64 {
	return []float64{1.0}
}
