package LinAlg

import "testing"

func Test_MatrixVectorMultiplication(t *testing.T) {
	// Arrange
	v := MakeVector([]float64{1, 2})
	m := MakeMatrix(1, 2, []float64{3, 4})

	// Act
	r := m.MultVector(&v)

	// Assert
	if r.Size() != 1 {
		t.Errorf("Resulting vector must have size 1, but is %d", r.Size())
	}
	if expected := float64(11); floatEquals(r.Get(0), expected, EPSILON) == false {
		t.Errorf("Matrix-Vector multiplication error, %f != %f", expected, r.Get(0))
	}
}

func Test_MatrixVectorMultiplication2(t *testing.T) {
	// Arrange
	v := MakeVector([]float64{1, 2})
	m := MakeMatrix(2, 2, []float64{3, 4, -2, -9})

	// Act
	r := m.MultVector(&v)

	// Assert
	if r.Size() != 2 {
		t.Errorf("Resulting vector must have size 2, but is %d", r.Size())
	}
	if expected := float64(11); floatEquals(r.Get(0), expected, EPSILON) == false {
		t.Errorf("Matrix-Vector multiplication error, %f != %f", expected, r.Get(0))
	}
	if expected := float64(16); floatEquals(r.Get(1), expected, EPSILON) == false {
		t.Errorf("Matrix-Vector multiplication error, %f != %f", expected, r.Get(1))
	}
}
