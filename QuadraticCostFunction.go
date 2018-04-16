package main

import (
	"SimpleNeuralNet/MNISTImport"
	"fmt"
)

type QuadtraticCostFunction struct{}

// -- Stringer --

func (QuadtraticCostFunction) String() string {
	return "Quadratic"
}

// -- CostFunction --

func (QuadtraticCostFunction) Evaluate(network *Network, trainingSamples []MNISTImport.TrainingSample) float64 {
	var cost float64
	mb := CreateMiniBatch(network.nNodes(), network.nWeights())
	for _, x := range trainingSamples {
		network.SetInputActivations(x.InputActivations, &mb)
		network.Feedforward(&mb)
		a := network.GetOutputLayerActivations(&mb)
		diff := GetError(x.OutputActivations, a)
		cost += diff * diff
	}
	cost /= float64(2 * len(trainingSamples))
	return cost
}

func caculateDeltaCost(j int, layer int, n *Network, mb *Minibatch, ts *MNISTImport.TrainingSample) float64 {
	if layer == n.getOutputLayerIndex() {
		a_i := n.GetActivation(j, layer, mb)
		dCda := a_i
		dCda -= ts.OutputActivations[j]
		z_i := n.CalculateZ(j, layer, mb)
		ds := SigmoidPrime(z_i)
		delta := dCda * ds
		return delta
	}
	nNextNodes := n.nNodesInLayer(layer + 1)
	var tmp float64
	for k := 0; k < nNextNodes; k++ {
		weight_kj := n.GetWeight(k, j, layer+1)
		delta_k := caculateDeltaCost(k, layer+1, n, mb, ts)
		tmp += weight_kj * delta_k
	}
	z_j := n.CalculateZ(j, layer, mb)
	s := SigmoidPrime(z_j)
	delta := tmp * s
	return delta
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
		delta_j := caculateDeltaCost(j, layer, network, &mb, &x)
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
		delta_j := caculateDeltaCost(j, layer, network, &mb, &x)
		dCdw += a_k * delta_j
	}
	dCdw /= float64(len(trainingSamples))
	return dCdw
}

func (QuadtraticCostFunction) CalculateErrorInOutputLayer(n *Network, outputActivations []float64, mb *Minibatch) {
	// Equation (BP1) and (30), Chapter 2 of http://neuralnetworksanddeeplearning.com
	outputLayerIdx := n.getOutputLayerIndex()
	nNodes := n.nNodesInLayer(outputLayerIdx)
	for i := 0; i < nNodes; i++ {
		a_i := n.GetActivation(i, outputLayerIdx, mb)
		dCda := a_i
		dCda -= outputActivations[i]
		z_i := n.CalculateZ(i, outputLayerIdx, mb)
		ds := SigmoidPrime(z_i)
		delta := dCda * ds
		n.SetDelta(delta, i, outputLayerIdx, mb)
	}
}
