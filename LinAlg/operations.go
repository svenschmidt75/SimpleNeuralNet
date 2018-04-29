package LinAlg

import "fmt"

func binopVectors(v1 *Vector, v2 *Vector, binop func(float64, float64) float64) *Vector {
	if v1.Size() != v2.Size() {
		panic(fmt.Sprintf("LinAlg.AddVectors: Vector sizes %d and %d must be the same", v1.Size(), v2.Size()))
	}
	result := MakeEmptyVector(v1.Size())
	for i := 0; i < v1.Size(); i++ {
		e1 := v1.Get(i)
		e2 := v2.Get(i)
		r := binop(e1, e2)
		result.Set(i, r)
	}
	return result
}

func AddVectors(v1 *Vector, v2 *Vector) *Vector {
	if v1.Size() != v2.Size() {
		panic(fmt.Sprintf("LinAlg.AddVectors: Vector sizes %d and %d must be the same", v1.Size(), v2.Size()))
	}
	return binopVectors(v1, v2, func(e1 float64, e2 float64) float64 {
		return e1 + e2
	})
}

func SubtractVectors(v1 *Vector, v2 *Vector) *Vector {
	if v1.Size() != v2.Size() {
		panic(fmt.Sprintf("LinAlg.AddVectors: Vector sizes %d and %d must be the same", v1.Size(), v2.Size()))
	}
	return binopVectors(v1, v2, func(e1 float64, e2 float64) float64 {
		return e1 - e2
	})
}

func OuterProduct(v1 *Vector, v2 *Vector) *Matrix {
	if v1.Size() != v2.Size() {
		panic(fmt.Sprintf("LinAlg.OuterProduct: Vector sizes %d and %d must be the same", v1.Size(), v2.Size()))
	}
	m := MakeEmptyMatrix(v1.Size(), v1.Size())
	for row := 0; row < v1.Size(); row++ {
		for col := 0; col < v1.Size(); col++ {
			value := v1.Get(row) * v2.Get(col)
			m.Set(row, col, value)
		}
	}
	return m
}
