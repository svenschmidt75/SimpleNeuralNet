package main

import "math"

type Network struct {
	nLayers []int
}

func sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}
