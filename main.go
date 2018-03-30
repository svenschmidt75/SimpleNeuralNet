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
		network := CreateNetwork([]int{28 * 28, 100, 10})
		network.InitializeNetworkWeightsAndBiases()

		dataDir := "/home/svenschmidt75/Develop/Go/MNIST"
		fmt.Printf("Location of training files (%s): ", dataDir)
		userDataDir := ""
		fmt.Scanf("%s\n", &userDataDir)
		if userDataDir == "" {
			userDataDir = dataDir
		}
		fmt.Printf("Importing training data from %s...\n", userDataDir)
		totalDataSet := MNISTImport.ImportData(userDataDir, "train-images.idx3-ubyte", "train-labels.idx1-ubyte")
		fmt.Printf("Read %d train images\n", totalDataSet.Length())
		fmt.Printf("Importing test data from %s...\n", userDataDir)
		testData := MNISTImport.ImportData(userDataDir, "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
		fmt.Printf("Read %d test images\n", testData.Length())
		fmt.Printf("How many training samples to consider (max %d): ", totalDataSet.Length())
		nTrainingSamples := 1000
		fmt.Scanf("%d\n", &nTrainingSamples)
		if nTrainingSamples > totalDataSet.Length() {
			nTrainingSamples = totalDataSet.Length()
		}
		validationDataFraction := float32(0.1)
		fmt.Printf("Fraction of training data for validation (%f): ", validationDataFraction)
		fmt.Scanf("%f\n", &validationDataFraction)
		trainingData, validationData := totalDataSet.Split(validationDataFraction, nTrainingSamples)
		fmt.Printf("Generating %d training samples...\n", trainingData.Length())
		ts := trainingData.GenerateTrainingSamples(trainingData.Length())
		fmt.Printf("Generating %d validation samples...\n", validationData.Length())
		vs := validationData.GenerateTrainingSamples(validationData.Length())

		epochs := 10
		eta := float32(4)
		miniMatchSize := 10
		fmt.Print("#epochs: ")
		fmt.Scanf("%d\n", &epochs)
		fmt.Print("learning rate eta: ")
		fmt.Scanf("%f\n", &eta)
		fmt.Print("mini batch size: ")
		fmt.Scanf("%d\n", &miniMatchSize)
		fmt.Print("Training neural network...\n")
		network.Train(ts, vs, epochs, eta, miniMatchSize)

		// run against test data
		var correctPredications int
		mb := CreateMiniBatch(network.nNodes(), network.nWeights())
		fmt.Printf("\nGenerating %d training samples for test data...\n", testData.Length())
		ts = testData.GenerateTrainingSamples(testData.Length())
		for testIdx := range ts {
			network.SetInputActivations(ts[testIdx].InputActivations, &mb)
			network.Feedforward(&mb)
			idx := network.getNodeBaseIndex(network.getOutputLayerIndex())
			as := mb.a[idx:]
			err := GetError(ts[testIdx].ExpectedClass, as)
			predictionIndex := GetIndex(as)
			if ts[testIdx].ExpectedClass == predictionIndex {
				correctPredications++
			}
			fmt.Printf("Index %d: Error is %f. Predicted %d, is %d\n", testIdx, err, predictionIndex, testData.GetResult(testIdx))
		}
		fmt.Printf("%d/%d correct predication\n", correctPredications, testData.Length())
		fmt.Printf("Error rate: %f\n", float64(correctPredications)/float64(testData.Length()))

		filename := "./n.gob"
		fmt.Printf("Enter a filename to serialize the network to (%s): ", filename)
		fmt.Scanf("%s\n", &filename)
		if filename == "" {
			filename = "./n.gob"
		}
		fmt.Printf("\nSerializing network to %s...\n", filename)
		err := writeGob(filename, &network)
		if err != nil {
			fmt.Println(err)
		}
	} else if idx == 2 {
		filename := "./n.gob"
		fmt.Printf("Enter the network filename (%s): ", filename)
		fmt.Scanf("%s\n", &filename)
		if filename == "" {
			filename = "./n.gob"
		}
		fmt.Printf("Deserializing network from %s...\n", filename)
		network := new(Network)
		err := readGob(filename, network)
		if err != nil {
			fmt.Println(err)
		}
		dataDir := "/home/svenschmidt75/Develop/Go/MNIST"
		fmt.Printf("Location of test data (%s): ", dataDir)
		userDataDir := ""
		fmt.Scanf("%s\n", &userDataDir)
		if userDataDir == "" {
			userDataDir = dataDir
		}
		fmt.Printf("Importing test data from %s...\n", userDataDir)
		testData := MNISTImport.ImportData(userDataDir, "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
		fmt.Printf("Read %d test images\n", testData.Length())
		ts := testData.GenerateTrainingSamples(testData.Length())

		// run against test data
		var correctPredications int
		mb := CreateMiniBatch(network.nNodes(), network.nWeights())
		fmt.Printf("\nGenerating %d training samples for test data...\n", testData.Length())
		ts = testData.GenerateTrainingSamples(testData.Length())
		for testIdx := range ts {
			network.SetInputActivations(ts[testIdx].InputActivations, &mb)
			network.Feedforward(&mb)
			idx := network.getNodeBaseIndex(network.getOutputLayerIndex())
			as := mb.a[idx:]
			err := GetError(ts[testIdx].ExpectedClass, as)
			predictionIndex := GetIndex(as)
			if ts[testIdx].ExpectedClass == predictionIndex {
				correctPredications++
			}
			fmt.Printf("Index %d: Error is %f. Predicted %d, is %d\n", testIdx, err, predictionIndex, testData.GetResult(testIdx))
		}
		fmt.Printf("%d/%d correct predictions\n", correctPredications, testData.Length())
		fmt.Printf("Accuracy: %f\n", float64(correctPredications)/float64(testData.Length()))

	}
}
