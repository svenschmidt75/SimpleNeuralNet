package main

import (
	"encoding/gob"
	"io"
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

func GetError(outputActivations []float64, a []float64) float64 {
	var err float64
	for idx := range a {
		d1 := outputActivations[idx]
		d2 := a[idx]
		err += (d1 - d2) * (d1 - d2)
	}
	return math.Sqrt(err)
}

func GetClass(a []float64) int {
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

func WriteGobToFile(filePath string, object interface{}) error {
	file, err := os.Create(filePath)
	if err == nil {
		WriteGob(file, object)
	}
	file.Close()
	return err
}

func WriteGob(w io.Writer, object interface{}) error {
	encoder := gob.NewEncoder(w)
	err := encoder.Encode(object)
	return err
}

func ReadGobFromFile(filePath string, object interface{}) error {
	file, err := os.Open(filePath)
	if err == nil {
		err = ReadGob(file, object)
	}
	file.Close()
	return err
}

func ReadGob(r io.Reader, object interface{}) error {
	decoder := gob.NewDecoder(r)
	err := decoder.Decode(object)
	return err
}
