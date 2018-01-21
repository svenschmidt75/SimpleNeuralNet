package main

import (
	"fmt"
)

func main() {
	fmt.Println("Hello")
	network := CreateNetwork([]int{2, 3, 2})

	network.weights[0] = 1
	network.weights[1] = 2
	network.weights[2] = 3
	network.weights[3] = 4
	network.weights[4] = 5
	network.weights[5] = 6
	network.weights[6] = 7
	network.weights[7] = 8
	network.weights[8] = 9
	network.weights[9] = 10
	network.weights[10] = 11
	network.weights[11] = 12

	network.biases[0] = 1
	network.biases[1] = 2
	network.biases[2] = 3
	network.biases[3] = 4
	network.biases[4] = 5

	network.activations[0] = 1
	network.activations[1] = 2

	index := network.GetWeightIndex(0, 3, 1)
	network.activations[index] = 2


	network.Feedforward()
}
