package main

import (
	"SimpleNeuralNet/MNISTImport"
	"fmt"
)

func main() {
	fmt.Print("1. Train neural network with MNIST data\n")
	fmt.Print("2. Run neural network on MNIST test data\n")
	idx := 1
	fmt.Scanf("%d\n", &idx)

	if idx == 1 {
		network := CreateNetwork([]int{28 * 28, 30, 10})
		network.InitializeNetworkWeightsAndBiases()

		userDataDir := "/home/svenschmidt75/Develop/Go/MNIST"
		fmt.Printf("Importing training data from %s...\n", userDataDir)
		totalDataSet := MNISTImport.ImportData(userDataDir, "train-images.idx3-ubyte", "train-labels.idx1-ubyte")
		fmt.Printf("Read %d train images\n", totalDataSet.Length())
		fmt.Printf("Importing test data from %s...\n", userDataDir)
		testData := MNISTImport.ImportData(userDataDir, "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
		fmt.Printf("Read %d test images\n", testData.Length())
		nTrainingSamples := 60000
		validationDataFraction := float32(0.1)
		trainingData, validationData := totalDataSet.Split(validationDataFraction, nTrainingSamples)
		fmt.Printf("Generating %d training samples...\n", trainingData.Length())
		ts := trainingData.GenerateTrainingSamples(trainingData.Length())
		fmt.Printf("Generating %d validation samples...\n", validationData.Length())
		vs := validationData.GenerateTrainingSamples(validationData.Length())

		epochs := 30
		eta := float32(0.5)
		miniMatchSize := 10
		network.Train(ts, vs, epochs, eta, miniMatchSize, CrossEntropyCostFunction{})

		fmt.Printf("\nGenerating %d training samples for test data...\n", testData.Length())
		ts = testData.GenerateTrainingSamples(testData.Length())

		// run against test data
		accuracy := network.RunSamples(ts)
		fmt.Printf("Accuracy: %f\n", accuracy)

		filename := "./n.gob"
		fmt.Printf("\nSerializing network to %s...\n", filename)
		err := WriteGobToFile(filename, &network)
		if err != nil {
			fmt.Println(err)
		}
	} else if idx == 2 {
		filename := "./n.gob"
		fmt.Printf("Deserializing network from %s...\n", filename)
		network := new(Network)
		err := ReadGobFromFile(filename, network)
		if err != nil {
			fmt.Println(err)
		}
		userDataDir := "/home/svenschmidt75/Develop/Go/MNIST"
		fmt.Printf("Importing test data from %s...\n", userDataDir)
		testData := MNISTImport.ImportData(userDataDir, "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
		fmt.Printf("Read %d test images\n", testData.Length())
		ts := testData.GenerateTrainingSamples(testData.Length())

		// run against test data
		accuracy := network.RunSamples(ts)
		fmt.Printf("Accuracy: %f\n", accuracy)
	}
}
