package LinAlg

import (
	"SimpleNeuralNet/Utility"
	"bytes"
	"testing"
)

func Test_Dotproduct(t *testing.T) {
	// Arrange
	v1 := MakeVector([]float64{1, 2})
	v2 := MakeVector([]float64{3, 4})

	// Act
	dp := v1.DotProduct(v2)

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
	r := v1.Hadamard(v2)

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
	v = v.F(func(v float64) float64 {
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

func TestVectorSerialization(t *testing.T) {
	// Arrange
	v1 := MakeVector([]float64{1, 2})

	// Act
	var buf bytes.Buffer
	err := Utility.WriteGob(&buf, v1)
	if err != nil {
		t.Errorf("Error serializing vector")
	}

	v2 := new(Vector)
	err = Utility.ReadGob(&buf, v2)
	if err != nil {
		t.Error("Error deserializing vector")
	}

	// Assert
	if expected := v1.Size(); v2.Size() != expected {
		t.Errorf("Vector size must be %d, but is %d", expected, v2.Size())
	}
	if expected := v1.Get(0); floatEquals(v2.Get(0), expected, EPSILON) == false {
		t.Errorf("Serialization error, %f != %f", expected, v2.Get(0))
	}
	if expected := v1.Get(1); floatEquals(v2.Get(1), expected, EPSILON) == false {
		t.Errorf("Serialization error, %f != %f", expected, v2.Get(1))
	}
}
