package LinAlg

import "fmt"

func AddVectors(v1 *Vector, v2 *Vector) *Vector {
	if v1.Size() != v2.Size() {
		panic(fmt.Sprintf("LinAlg.AddVectors: Vector sizes %d and %d must be the same", v1.Size(), v2.Size()))
	}
	result := MakeEmptyVector(v1.Size())
	for idx := range result.data {
		e1 := v1.data[idx]
		e2 := v2.data[idx]
		result.data[idx] = e1 + e2
	}
	return result
}

func SubtractVectors(v1 *Vector, v2 *Vector) *Vector {
	if v1.Size() != v2.Size() {
		panic(fmt.Sprintf("LinAlg.SubVectors: Vector sizes %d and %d must be the same", v1.Size(), v2.Size()))
	}
	result := MakeEmptyVector(v1.Size())
	for idx := range result.data {
		e1 := v1.data[idx]
		e2 := v2.data[idx]
		result.data[idx] = e1 - e2
	}
	return result
}

func OuterProduct(v1 *Vector, v2 *Vector) *Matrix {
	m := MakeEmptyMatrix(v1.Size(), v2.Size())
	for row := 0; row < v1.Size(); row++ {
		for col := 0; col < v2.Size(); col++ {
			value := v1.Get(row) * v2.Get(col)
			m.Set(row, col, value)
		}
	}
	return m
}
