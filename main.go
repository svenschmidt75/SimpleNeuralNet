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
		trainingData := MNISTImport.ImportData(userDataDir, "train-images.idx3-ubyte", "train-labels.idx1-ubyte")
		fmt.Printf("Read %d train images\n", trainingData.Length())
		fmt.Printf("Importing test data from %s...\n", userDataDir)
		testData := MNISTImport.ImportData(userDataDir, "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
		fmt.Printf("Read %d test images\n", testData.Length())

		fmt.Printf("How many training samples to train on (max %d): ", trainingData.Length())
		nTrainingSamples := 100
		fmt.Scanf("%d", &nTrainingSamples)
		if nTrainingSamples > trainingData.Length() {
			nTrainingSamples = trainingData.Length()
		}
		fmt.Printf("\nGenerating %d training samples...\n", nTrainingSamples)
		ts := trainingData.GenerateTrainingSamples(nTrainingSamples)

		epochs := 2
		eta := float32(0.5)
		miniMatchSize := 10
		fmt.Print("#epochs: ")
		fmt.Scanf("%d", &epochs)
		fmt.Print("learning rate eta: ")
		fmt.Scanf("%f", &eta)
		fmt.Print("mini batch size: ",)
		fmt.Scanf("%d", &miniMatchSize)
		fmt.Print("Training neural network...\n")
		network.Train(ts, epochs, eta, miniMatchSize)

		// run against test data
		var correctPredications int
		mb := CreateMiniBatch(network.nNodes(), network.nWeights())
		fmt.Printf("\nGenerating %d training samples for test data...\n", testData.Length())
		ts = testData.GenerateTrainingSamples(nTrainingSamples)
		for testIdx := range ts {
			network.SetInputActivations(ts[testIdx].InputActivations, &mb)
			network.Feedforward(&mb)
			idx := network.getNodeBaseIndex(2)
			as := mb.a[idx:]
			err := GetError(ts[testIdx].OutputActivations, as)
			predictionIndex := GetIndex(as)
			if ts[testIdx].OutputActivations[predictionIndex] == 1 {
				correctPredications++
			}
			fmt.Printf("Index %d: Error is %f. Predicted %d, is %d\n", testIdx, err, predictionIndex, testData.GetResult(testIdx))
		}
		fmt.Printf("%d/%d correct predication\n", correctPredications, testData.Length())
		fmt.Printf("Error rate: %f\n", 1.0 - float64(correctPredications) / float64(testData.Length()))
	}
}
