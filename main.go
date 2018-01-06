package main

import (
	"fmt"
)

func main() {
	fmt.Println("Hello")
	network := CreateNetwork([]int{2, 3, 2})

	fmt.Printf("%#v", network)

	ai := network.GetWeightIndex(1, 2, 1)
	fmt.Printf("%#v", ai)
}
