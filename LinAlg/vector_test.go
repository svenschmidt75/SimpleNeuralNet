package LinAlg

import "testing"

func Test_Hadamard(t *testing.T) {
	// Arrange
	v1 := MakeVector([]float64{1, 2})
	v2 := MakeVector([]float64{-5, -9})

	// Act
	r := v1.Hadamard(&v2)

	// Assert
	if r.Size() != 2 {
		t.Errorf("Resulting vector must have size 2, but is %d", r.Size())
	}
	if expected := float64(-5); floatEquals(r.Get(0), expected, EPSILON) == false {
		t.Errorf("Hadamard error, %f != %f", expected, r.Get(0))
	}
	if expected := float64(-18); floatEquals(r.Get(1), expected, EPSILON) == false {
		t.Errorf("Hadamard error, %f != %f", expected, r.Get(1))
	}
}
