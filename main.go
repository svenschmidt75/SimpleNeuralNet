package main

import (
	"fmt"
)

func main() {
	fmt.Println("Hello")
	network := CreateNetwork([]int{1, 1, 1})

	fmt.Printf("%#v", network)
}
