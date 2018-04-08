package main

import (
	"SimpleNeuralNet/MNISTImport"
	"fmt"
)

type QuadtraticCostFunction struct{}

func (QuadtraticCostFunction) Evaluate(network *Network, trainingSamples []MNISTImport.TrainingSample) float64 {
	var cost float64
	mb := CreateMiniBatch(network.nNodes(), network.nWeights())
	for _, x := range trainingSamples {
		network.SetInputActivations(x.InputActivations, &mb)
		network.Feedforward(&mb)
		a := network.GetOutputLayerActivations(&mb)
		diff := GetError(x.ExpectedClass, a)
		cost += diff * diff
	}
	cost /= float64(2 * len(trainingSamples))
	return cost
}

func (QuadtraticCostFunction) GradBias(j int, layer int, network *Network, trainingSamples []MNISTImport.TrainingSample) float64 {
	if layer == 0 {
		panic(fmt.Sprintf("Layer must be > 0"))
	}
	var delta float64
	mb := CreateMiniBatch(network.nNodes(), network.nWeights())
	for _, x := range trainingSamples {
		network.SetInputActivations(x.InputActivations, &mb)
		network.Feedforward(&mb)
		delta_j := network.CaculateDelta(j, layer, &mb, &x)
		delta += delta_j
	}
	delta /= float64(len(trainingSamples))
	return delta
}

func (QuadtraticCostFunction) GradWeight(j int, k int, layer int, network *Network, trainingSamples []MNISTImport.TrainingSample) float64 {
	if layer == 0 {
		panic(fmt.Sprintf("Layer must be > 0"))
	}
	var dCdw float64
	mb := CreateMiniBatch(network.nNodes(), network.nWeights())
	for _, x := range trainingSamples {
		network.SetInputActivations(x.InputActivations, &mb)
		network.Feedforward(&mb)
		a_k := network.GetActivation(k, layer-1, &mb)
		delta_j := network.CaculateDelta(j, layer, &mb, &x)
		dCdw += a_k * delta_j
	}
	dCdw /= float64(len(trainingSamples))
	return dCdw
}
