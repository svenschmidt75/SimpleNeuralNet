package MNISTImport

import "testing"

func TestImportImageFile(t *testing.T) {
	ImportImageFile("/home/svenschmidt75/Develop/Go/MNIST/train-images.idx3-ubyte")
}

func TestImportLabelFile(t *testing.T) {
	ImportLabelFile("/home/svenschmidt75/Develop/Go/MNIST/train-labels.idx1-ubyte")
}
