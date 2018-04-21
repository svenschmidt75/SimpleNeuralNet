package LinAlg

type Vector struct {
	data []float64
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

func (v *Vector) Set(index int, value float64) {
	v.data[index] = value
}

func (v *Vector) Get(index int) float64 {
	return v.data[index]
}
