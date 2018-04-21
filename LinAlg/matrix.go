package LinAlg

import "fmt"

type Matrix struct {
	Rows int
	Cols int
	data []float64
}

func MakeMatrix(rows int, cols int, data []float64) Matrix {
	if size := rows * cols; size != len(data) {
		panic(fmt.Sprintf("LinAlg.Matrix.MakeMatrix: Matrix data has size %d, but %d expected", len(data), size))
	}
	return Matrix{Rows: rows, Cols: cols, data: data}
}

func MakeEmptyMatrix(rows int, cols int) Matrix {
	size := rows * cols
	return Matrix{Rows: rows, Cols: cols, data: make([]float64, size)}
}

func (m *Matrix) index(row int, col int) int {
	return row*m.Cols + col
}

func (m *Matrix) Set(row int, col int, value float64) {
	idx := m.index(row, col)
	m.data[idx] = value
}

func (m *Matrix) Get(row int, col int) float64 {
	idx := m.index(row, col)
	return m.data[idx]
}

func (m *Matrix) Transpose() Matrix {
	return Matrix{}
}

func (m *Matrix) MultVector(v *Vector) Vector {
	if m.Cols != v.Size() {
		panic(fmt.Sprintf("LinAlg.Matrix.MultVector: Matrix number of columns %d must equal vector size %d", m.Cols, v.Size()))
	}
	result := MakeEmptyVector(m.Rows)
	for row := 0; row < m.Rows; row++ {
		var value float64
		for col := 0; col < m.Cols; col++ {
			value += m.Get(row, col) * v.Get(col)
		}
		result.Set(row, value)
	}
	return result
}

func (m *Matrix) Scalar(scalar float64) {
	for idx := range m.data {
		m.data[idx] *= scalar
	}
}
