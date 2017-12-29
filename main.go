package main

import (
	"fmt"
)

func main() {
	fmt.Println("Hello")
	nLayers := [...]int{1, 1, 1}
	network := Network{nLayers: nLayers[0:len(nLayers)]}

	fmt.Println("%v", network)
}
