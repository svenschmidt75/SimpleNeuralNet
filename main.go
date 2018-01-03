package main

import (
	"fmt"
)

func main() {
	fmt.Println("Hello")
	layers := [...]int{1, 1, 1}
	network := Network{nLayers: len(layers), layers: layers[0:]}

	fmt.Println("%v", network)
}
