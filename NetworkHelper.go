package main

import (
	"encoding/gob"
	"math"
	"math/rand"
	"os"
)

func max(lhs int, rhs int) int {
	if lhs < rhs {
		return rhs
	}
	return lhs
}

func min(lhs int, rhs int) int {
	if lhs < rhs {
		return lhs
	}
	return rhs
}

func GenerateRandomIndices(size int) []int {
	// generate random permutation
	perm := rand.Perm(size)
	return perm
}

func GetError(predictedClass int, a []float64) float64 {
	var err float64
	for idx := range a {
		var d1 float64 = 0
		if idx == predictedClass {
			d1 = 1
		}
		d2 := a[idx]
		err += (d1 - d2) * (d1 - d2)
	}
	return math.Sqrt(err)
}

func GetIndex(a []float64) int {
	var index int
	var value float64 = -1
	for idx := range a {
		if a[idx] > value {
			value = a[idx]
			index = idx
		}

	}
	return index
}

func WriteGob(filePath string, object interface{}) error {
	file, err := os.Create(filePath)
	if err == nil {
		encoder := gob.NewEncoder(file)
		encoder.Encode(object)
	}
	file.Close()
	return err
}

func ReadGob(filePath string, object interface{}) error {
	file, err := os.Open(filePath)
	if err == nil {
		decoder := gob.NewDecoder(file)
		err = decoder.Decode(object)
	}
	file.Close()
	return err
}
