package main

import (
	"math"
)

type Network struct {
	layers      []int
	activations []float64
	biases      []float64
	weights     []float64
}

func sum(xs []int) int {
	sum := 0
	for _, i := range xs {
		sum += i
	}
	return sum
}

func getNumberOfWeights(xs []int) int {
	n := 0
	x1 := xs[0]
	for i := 1; i < len(xs); i++ {
		x2 := xs[i]
		n += x1 * x2
		x1 = x2
	}
	return n
}

func CreateNetwork(layers []int) Network {
	nActivations := sum(layers)
	nBiases := sum(layers[1:])
	nWeights := getNumberOfWeights(layers)
	return Network{layers: layers, activations: make([]float64, nActivations), biases: make([]float64, nBiases), weights: make([]float64, nWeights)}
}

func Sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Exp(-z))
}

func (n Network) GetActivationBaseIndex(layer int) int {
	bi := 0
	for _, i := range n.layers {
		if i >= layer {
			break
		}
		bi += i
	}
	return bi
}

func (n Network) GetActivation(index int, layer int) *float64 {
	bi := n.GetActivationBaseIndex(layer)
	return &n.activations[bi+index]
}

func (n Network) GetBiasBaseIndex(layer int) int {
	bi := 0
	for _, i := range n.layers {
		if i == 0 {
			continue
		}
		if i >= layer {
			break
		}
		bi += i
	}
	return bi
}

func (n Network) GetBias(index int, layer int) *float64 {
	bi := n.GetBiasBaseIndex(layer)
	return &n.biases[bi+index]
}

func (n Network) GetWeightBaseIndex(layer int) int {
	return getNumberOfWeights(n.layers[0:layer])
}

func (n Network) GetWeight(i int, j int, layer int) *float64 {
	bi := n.GetWeightBaseIndex(layer)

}

func (n Network) Feedforward() []float64 {

	return []float64{1.0}
}
