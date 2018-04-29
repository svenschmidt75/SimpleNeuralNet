package LinAlg

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"math"
)

type Vector struct {
	data []float64
}

//
// Implement interface 'GobEncoder'
//
func (v *Vector) GobEncode() ([]byte, error) {
	w := new(bytes.Buffer)
	encoder := gob.NewEncoder(w)
	err := encoder.Encode(v.data)
	if err != nil {
		return nil, err
	}
	return w.Bytes(), nil
}

//
// Implement interface 'GobDecoder'
//
func (v *Vector) GobDecode(buf []byte) error {
	r := bytes.NewBuffer(buf)
	decoder := gob.NewDecoder(r)
	return decoder.Decode(&v.data)
}

func MakeVector(data []float64) Vector {
	return Vector{data: data}
}

func MakeEmptyVector(size int) Vector {
	return Vector{data: make([]float64, size)}
}

func (v *Vector) Size() int {
	return len(v.data)
}

func (v Vector) Set(index int, value float64) {
	v.data[index] = value
}

func (v *Vector) Get(index int) float64 {
	return v.data[index]
}

func (v1 *Vector) DotProduct(v2 Vector) float64 {
	if v1.Size() != v2.Size() {
		panic(fmt.Sprintf("LinAlg.Vector.DotProduct: Vector sizes %d and %d must be the same", v1.Size(), v2.Size()))
	}
	var d float64 = 0
	for i := 0; i < v1.Size(); i++ {
		e1 := v1.Get(i)
		e2 := v2.Get(i)
		d += e1 * e2
	}
	return d
}

func (v *Vector) Add(other Vector) {
	if v.Size() != other.Size() {
		panic(fmt.Sprintf("LinAlg.Vector.Add: Vector sizes %d and %d must be the same", v.Size(), other.Size()))
	}
	for i := 0; i < v.Size(); i++ {
		e1 := v.Get(i)
		e2 := other.Get(i)
		d := e1 * e2
		v.Set(i, d)
	}
}

func (v *Vector) ScalarMultiplication(scalar float64) {
	for idx := range v.data {
		v.data[idx] *= scalar
	}
}

func (v *Vector) F(f func(float64) float64) Vector {
	result := MakeEmptyVector(v.Size())
	for idx := range v.data {
		value := v.data[idx]
		value = f(value)
		result.data[idx] = value
	}
	return result
}

func (v *Vector) Hadamard(other *Vector) Vector {
	if v.Size() != other.Size() {
		panic(fmt.Sprintf("LinAlg.Vector.Hadamard: Vectors must have same size, but is %d and %d", v.Size(), other.Size()))
	}
	result := MakeEmptyVector(v.Size())
	for idx := range v.data {
		e1 := v.data[idx]
		e2 := other.data[idx]
		result.Set(idx, e1*e2)
	}
	return result
}

func (v *Vector) EuklideanNorm() float64 {
	var err float64
	for idx := range v.data {
		e := v.Get(idx)
		err += e * e
	}
	return math.Sqrt(err)
}
