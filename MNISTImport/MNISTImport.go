package MNISTImport

import (
	"encoding/binary"
	"image"
	"image/color"
	"io/ioutil"
)

type MNISTData struct {
	// all training samples
	trainingInputActivations [][]float64

	// labels for each training sample
	trainingExpectedResult []byte

	// all test samples
	testInputActivations [][]float64

	// labels for each test sample
	testExpectedResult []byte
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
		//outputFile, err := os.Create(fmt.Sprintf("/home/svenschmidt75/Downloads/MNISTDecoded/test%d.png", imageIdx))
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

func Import(dir string) MNISTData {
	var output MNISTData
	output.trainingInputActivations = ImportImageFile(dir + "train-images.idx3-ubyte")
	output.trainingExpectedResult = ImportLabelFile(dir + "train-labels.idx1-ubyte")
	output.testInputActivations = ImportImageFile(dir + "t10k-images.idx3-ubyte")
	output.testExpectedResult = ImportLabelFile(dir + "t10k-labels.idx1-ubyte")
	return output
}

func (m *MNISTData) GenerateTrainingSamples() []TrainingSample {
	tss := make([]TrainingSample, len(m.trainingInputActivations))
	for idx := range m.trainingInputActivations {
		ts := &tss[idx]
		ts.InputActivations = m.trainingInputActivations[idx]
		ts.OutputActivations = make([]float64, 10)
		digit := m.trainingExpectedResult[idx]
		ts.OutputActivations[digit] = 1
	}
	return tss
}
