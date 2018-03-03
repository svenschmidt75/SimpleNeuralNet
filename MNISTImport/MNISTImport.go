package MNISTImport

import (
	"io/ioutil"
	"encoding/binary"
	//"image"
	//"image/color"
	//"os"
	//"image/png"
	//"fmt"
)


type MNISTData struct {
	// all training samples
	trainingSamples [][]float64

	// labels for each training sample
	trainingInputActivations [][]float64

	// all test samples
	testSamples [][]float64

	// labels for each test sample
	testInputActivations [][]float64
}

func BuildFromImageFile(nImages int, nRows int, nCols int, data []byte) [][]float64 {
	output := make([][]float64, nImages)
	idx := 0
	for imageIdx := 0; imageIdx < nImages; imageIdx++ {
		img := make([]float64, nRows * nCols)
		output[imageIdx] = img
//		m := image.NewRGBA(image.Rect(0, 0, nCols, nRows))
		for rowIdx := 0; rowIdx < nRows; rowIdx++ {
			for colIdx := 0; colIdx < nRows; colIdx++ {
				value := data[idx]
				idx++
				img[rowIdx * nCols + colIdx] = float64(value) / 255
//				m.Set(colIdx, rowIdx, color.RGBA{value, 0, 0, 255})
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
