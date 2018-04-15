package MNISTImport

import (
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"io/ioutil"
	"math/rand"
	"path"
	"time"
)

type MNISTData struct {
	inputActivations [][]float64
	expectedResult   []byte
}

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

func BuildFromImageFile(nImages int, nRows int, nCols int, data []byte) [][]float64 {
	output := make([][]float64, nImages)
	idx := 0
	for imageIdx := 0; imageIdx < nImages; imageIdx++ {
		img := make([]float64, nRows*nCols)
		output[imageIdx] = img
		m := image.NewRGBA(image.Rect(0, 0, nCols, nRows))
		for rowIdx := 0; rowIdx < nRows; rowIdx++ {
			for colIdx := 0; colIdx < nRows; colIdx++ {
				value := data[idx]
				idx++
				img[rowIdx*nCols+colIdx] = float64(value) / 255
				m.Set(colIdx, rowIdx, color.RGBA{value, 0, 0, 255})
			}
		}
		//outputFile, err := os.Create(fmt.Sprintf("/home/svenschmidt75/Develop/Go/MNIST/TestImages/test%d.png", imageIdx))
		//if err != nil {
		//	panic("error")
		//}
		//png.Encode(outputFile, m)
		//outputFile.Close()
	}
	return output
}

func BuildFromLabelFile(nLabels int, data []byte) []byte {
	output := make([]byte, nLabels)
	for labelIdx := 0; labelIdx < nLabels; labelIdx++ {
		value := data[labelIdx]
		output[labelIdx] = value
	}
	return output
}

func ImportImageFile(fileName string) [][]float64 {
	data, err := ioutil.ReadFile(fileName)
	if err != nil {
		panic("Failed to read data")
	}
	magicNumber := binary.BigEndian.Uint32(data[0:4])
	if magicNumber != 0x0803 {
		panic("Image file format error")
	}
	nImages := int(binary.BigEndian.Uint32(data[4:8]))
	nRows := int(binary.BigEndian.Uint32(data[8:12]))
	nCols := int(binary.BigEndian.Uint32(data[12:16]))
	return BuildFromImageFile(nImages, nRows, nCols, data[16:])
}

func ImportLabelFile(fileName string) []byte {
	data, err := ioutil.ReadFile(fileName)
	if err != nil {
		panic("Failed to read data")
	}
	magicNumber := binary.BigEndian.Uint32(data[0:4])
	if magicNumber != 0x0801 {
		panic("Label file format error")
	}
	nLabels := int(binary.BigEndian.Uint32(data[4:8]))
	return BuildFromLabelFile(nLabels, data[8:])
}

func ImportData(dir string, imageFile string, labelFile string) MNISTData {
	var output MNISTData
	output.inputActivations = ImportImageFile(path.Join(dir, imageFile))
	output.expectedResult = ImportLabelFile(path.Join(dir, labelFile))
	return output
}

func (data MNISTData) Length() int {
	return len(data.inputActivations)
}

func (data MNISTData) GenerateTrainingSamples(length int) []TrainingSample {
	tss := make([]TrainingSample, length)
	for idx := range data.inputActivations {
		if idx >= length {
			break
		}
		ts := &tss[idx]
		ts.InputActivations = data.inputActivations[idx]
		ts.OutputActivations = make([]float64, 10)
		expectedResult := data.expectedResult[idx]
		ts.OutputActivations[expectedResult] = 1
	}
	return tss
}

func (data MNISTData) GetResult(index int) byte {
	return data.expectedResult[index]
}

func (data *MNISTData) Split(ratio float32, size int) (*MNISTData, *MNISTData) {
	if ratio <= 0 || ratio > 1 {
		panic(fmt.Sprintf("Ratio %f must be between (0,1]", ratio))
	}
	if size > data.Length() {
		panic(fmt.Sprintf("Training data size %d cannot be larger then the total data size %d", size, data.Length()))
	}
	totalSize := data.Length()
	perm := rand.Perm(totalSize)

	var GenerateData = func(size int, offset int) *MNISTData {
		newData := &MNISTData{make([][]float64, size), make([]byte, size)}
		for idx := 0; idx < size; idx++ {
			dataIdx := perm[offset+idx]
			newData.inputActivations[idx] = data.inputActivations[dataIdx]
			newData.expectedResult[idx] = data.expectedResult[dataIdx]
		}
		return newData
	}

	trainingSize := int(float32(size) * (1 - ratio))
	trainingData := GenerateData(trainingSize, 0)

	validationSize := size - trainingSize
	validationData := GenerateData(validationSize, trainingSize)

	return trainingData, validationData
}
