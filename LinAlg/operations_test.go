package LinAlg

import (
	"math"
	"testing"
)

const (
	EPSILON = 0.00000001
)

func floatEquals(a, b float64, eps float64) bool {
	return math.Abs(a-b) < eps
}

func Test_AddVectors(t *testing.T) {
	// Arrange
	v1 := MakeVector([]float64{1, 2})
	v2 := MakeVector([]float64{3, 4})

	// Act
	r := AddVectors(&v1, &v2)

	// Assert
	if expected := float64(4); floatEquals(r.Get(0), expected, EPSILON) == false {
		t.Error("Vector addition error")
	}
	if expected := float64(6); floatEquals(r.Get(1), expected, EPSILON) == false {
		t.Error("Vector addition error")
	}
}

func Test_SubtractVectors(t *testing.T) {
	// Arrange
	v1 := MakeVector([]float64{1, 2})
	v2 := MakeVector([]float64{3, 4})

	// Act
	r := SubtractVectors(&v1, &v2)

	// Assert
	if expected := float64(-2); floatEquals(r.Get(0), expected, EPSILON) == false {
		t.Error("Vector subtraction error")
	}
	if expected := float64(-2); floatEquals(r.Get(1), expected, EPSILON) == false {
		t.Error("Vector subtraction error")
	}
}
