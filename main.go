package main

import (
	"fmt"
	"SimpleNeuralNet/MNISTImport"
)

func main() {
	fmt.Print("1. Train neural network with MNIST data\n")
	fmt.Print("2. Run neural network on MNIST test data\n")
	idx := 1
	fmt.Scanf("%d", &idx)

	if idx == 1 {
		network := CreateNetwork([]int{28 * 28, 100, 10})
		network.InitializeNetworkWeightsAndBiases()

		dataDir := "/home/svenschmidt75/Develop/Go/MNIST"
		fmt.Printf("Location of training files (%s): ", dataDir)
		userDataDir := ""
		fmt.Scanf("%s", &userDataDir)
		if userDataDir == "" {
			userDataDir = dataDir
		}
		fmt.Printf("Importing training data from %s...\n", userDataDir)
		trainingData := MNISTImport.ImportData(userDataDir, "train-images.idx3-ubyte", "train-labels.idx1" +
			"-ubyte")
		fmt.Printf("Read %d train images\n", trainingData.Length())
		fmt.Printf("Importing test data from %s...\n", userDataDir)
//		testData := MNISTImport.ImportData(userDataDir, "t10k-images.idx3-ubyte", "t10k-labels.idx3-ubyte")

		fmt.Printf("How many training samples to train on (max %d): ", trainingData.Length())
		nTrainingSamples := 1000
		fmt.Scanf("%d", nTrainingSamples)
		if nTrainingSamples > trainingData.Length() {
			nTrainingSamples = trainingData.Length()
		}

		ts := trainingData.GenerateTrainingSamples(nTrainingSamples)


		network.Train(ts, 2, 0.5, 10)



	}













	fmt.Printf("%d", idx)
}
