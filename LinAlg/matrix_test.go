package LinAlg

import (
	"SimpleNeuralNet/Utility"
	"bytes"
	"testing"
)

func Test_MatrixVectorMultiplication(t *testing.T) {
	// Arrange
	v := MakeVector([]float64{1, 2})
	m := MakeMatrix(1, 2, []float64{3, 4})

	// Act
	r := m.Ax(v)

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
	r := m.Ax(v)

	// Assert
	if r.Size() != 2 {
		t.Errorf("Resulting vector must have size 2, but is %d", r.Size())
	}
	if expected := float64(11); floatEquals(r.Get(0), expected, EPSILON) == false {
		t.Errorf("Matrix-Vector multiplication error, %f != %f", expected, r.Get(0))
	}
	if expected := float64(-20); floatEquals(r.Get(1), expected, EPSILON) == false {
		t.Errorf("Matrix-Vector multiplication error, %f != %f", expected, r.Get(1))
	}
}

func Test_MatrixTranspose(t *testing.T) {
	// Arrange
	m := MakeMatrix(3, 2, []float64{3, 4, -2, -9, 4, 7})

	// Act
	mt := m.Transpose()

	// Assert
	if rows := mt.Rows; rows != 2 {
		t.Errorf("Resulting matrix must have %d rows, but is %d", 2, rows)
	}
	if cols := mt.Cols; cols != 3 {
		t.Errorf("Resulting matrix must have %d columns, but is %d", 3, cols)
	}
	if expected := float64(3); floatEquals(mt.Get(0, 0), expected, EPSILON) == false {
		t.Errorf("Matrix transpose error, %f != %f", expected, mt.Get(0, 0))
	}
	if expected := float64(-2); floatEquals(mt.Get(0, 1), expected, EPSILON) == false {
		t.Errorf("Matrix transpose error, %f != %f", expected, mt.Get(0, 1))
	}
	if expected := float64(4); floatEquals(mt.Get(0, 2), expected, EPSILON) == false {
		t.Errorf("Matrix transpose error, %f != %f", expected, mt.Get(0, 2))
	}
	if expected := float64(4); floatEquals(mt.Get(1, 0), expected, EPSILON) == false {
		t.Errorf("Matrix transpose error, %f != %f", expected, mt.Get(1, 0))
	}
	if expected := float64(-9); floatEquals(mt.Get(1, 1), expected, EPSILON) == false {
		t.Errorf("Matrix transpose error, %f != %f", expected, mt.Get(1, 1))
	}
	if expected := float64(7); floatEquals(mt.Get(1, 2), expected, EPSILON) == false {
		t.Errorf("Matrix transpose error, %f != %f", expected, mt.Get(1, 2))
	}
}

func TestMatrixSerialization(t *testing.T) {
	// Arrange
	m1 := MakeMatrix(3, 2, []float64{3, 4, -2, -9, 4, 7})

	// Act
	var buf bytes.Buffer
	err := Utility.WriteGob(&buf, m1)
	if err != nil {
		t.Errorf("Error serializing matrix")
	}

	m2 := new(Matrix)
	err = Utility.ReadGob(&buf, m2)
	if err != nil {
		t.Error("Error deserializing matrix")
	}

	// Assert
	if expected := m1.Rows; m2.Rows != expected {
		t.Errorf("Matrix must have %d rows, but is %d", expected, m2.Rows)
	}
	if expected := m1.Cols; expected != m2.Cols {
		t.Errorf("Matrix must have %d columns, but is %d", expected, m2.Cols)
	}
	if expected := m1.Get(0, 0); floatEquals(m2.Get(0, 0), expected, EPSILON) == false {
		t.Errorf("Matrix deserializing error, %f != %f", expected, m2.Get(0, 0))
	}
	if expected := m1.Get(0, 1); floatEquals(m2.Get(0, 1), expected, EPSILON) == false {
		t.Errorf("Matrix deserializing error, %f != %f", expected, m2.Get(0, 1))
	}
	if expected := m1.Get(0, 2); floatEquals(m2.Get(0, 2), expected, EPSILON) == false {
		t.Errorf("Matrix deserializing error, %f != %f", expected, m2.Get(0, 2))
	}
	if expected := m1.Get(1, 0); floatEquals(m2.Get(1, 0), expected, EPSILON) == false {
		t.Errorf("Matrix deserializing error, %f != %f", expected, m2.Get(1, 0))
	}
	if expected := m1.Get(1, 1); floatEquals(m2.Get(1, 1), expected, EPSILON) == false {
		t.Errorf("Matrix deserializing error, %f != %f", expected, m2.Get(1, 1))
	}
	if expected := m1.Get(1, 2); floatEquals(m2.Get(1, 2), expected, EPSILON) == false {
		t.Errorf("Matrix deserializing error, %f != %f", expected, m2.Get(1, 2))
	}
}
