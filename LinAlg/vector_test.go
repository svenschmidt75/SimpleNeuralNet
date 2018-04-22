package LinAlg

import "testing"

func Test_Dotproduct(t *testing.T) {
	// Arrange
	v1 := MakeVector([]float64{1, 2})
	v2 := MakeVector([]float64{3, 4})

	// Act
	dp := v1.DotProduct(&v2)

	// Assert
	var expected float64 = 11
	if floatEquals(dp, expected, EPSILON) == false {
		t.Error("Dot product error")
	}
}

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

func Test_F(t *testing.T) {
	// Arrange
	v := MakeVector([]float64{1, 2})

	// Act
	v.F(func(v float64) float64 {
		return v + 1
	})

	// Assert
	if expected := float64(2); floatEquals(v.Get(0), expected, EPSILON) == false {
		t.Errorf("F error, %f != %f", expected, v.Get(0))
	}
	if expected := float64(3); floatEquals(v.Get(1), expected, EPSILON) == false {
		t.Errorf("Hadamard error, %f != %f", expected, v.Get(1))
	}
}
