package neural

import (
	"errors"
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/goseventh/Phoenix-G/terminal"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func (nn *NeuralNet) Initialize(bootRandomly bool) (*mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense) {
	// Initialize biases/weights.
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	wHidden := mat.NewDense(nn.Config.InputNeurons, nn.Config.HiddenNeurons, nil)
	bHidden := mat.NewDense(1, nn.Config.HiddenNeurons, nil)
	wOut := mat.NewDense(nn.Config.HiddenNeurons, nn.Config.OutputNeurons, nil)
	bOut := mat.NewDense(1, nn.Config.OutputNeurons, nil)

	wHiddenRaw := wHidden.RawMatrix().Data
	bHiddenRaw := bHidden.RawMatrix().Data
	wOutRaw := wOut.RawMatrix().Data
	bOutRaw := bOut.RawMatrix().Data

	for _, param := range [][]float64{
		wHiddenRaw,
		bHiddenRaw,
		wOutRaw,
		bOutRaw,
	} {
		for i := range param {
			param[i] = randGen.Float64()

		}
	}
	// Define our neural network.
	nn.WHidden = wHidden
	nn.BHidden = bHidden
	nn.WOut = wOut
	nn.BOut = bOut

	// Define the output of the neural network.
	output := new(mat.Dense)
	return wHidden, bHidden, wOut, bOut, output

}

// Train trains a neural network using backpropagation.
// x,y = train
// z,b = predict
func (nn *NeuralNet) Train(x, y, z, b *mat.Dense) error {

	wHidden, bHidden, wOut, bOut, output :=
		nn.Initialize(true)

	//nn.SaveModel()
	// Use backpropagation to adjust the weights and biases.
	if err := nn.backpropagate(x, y, z, b, wHidden, bHidden, wOut, bOut, output); err != nil {
		return err
	}

	// Define our trained neural network.
	nn.WHidden = wHidden
	nn.BHidden = bHidden
	nn.WOut = wOut
	nn.BOut = bOut

	return nil
}

// train where a neural network left off using backpropagation.
// x,y = train
// z,b = predict
func (nn *NeuralNet) ContinueTraining(inputTrain, labelTrain, inputTest, labelTest *mat.Dense) error {
	//wHidden, bHidden, wOut, bOut, output :=
	//	nn.Initialize(false)

	// Use backpropagation to adjust the weights and biases.
	if err := nn.backpropagate(inputTrain, labelTrain, inputTest, labelTest, nn.WHidden, nn.BHidden, nn.WOut, nn.BOut, new(mat.Dense)); err != nil {
		return err
	}

	// Define our trained neural network.
	/*nn.WHidden = wHidden
	nn.BHidden = bHidden
	nn.WOut = wOut
	nn.BOut = bOut*/

	return nil
}

// backpropagate completes the backpropagation method.
// x,y = train
// z,b = predict
func (nn *NeuralNet) backpropagate(inputTrain, labelTrain, inputTest, labelTest, wHidden, bHidden, wOut, bOut, output *mat.Dense) error {

	log.Println("[cycle-ml] starting loop cycle")
	var acc float64
	var index int
	go func() {
		for {
			time.Sleep(time.Millisecond)
			fmt.Printf(terminal.LogColorGreen+"[cycle-ml]:"+
				terminal.LogColorWarning+"[%0.2f%s]complete"+
				terminal.LogColorHeader+" [%0.2f%s]accuracy "+
				terminal.LogColorWarning+"training in cycle: %v from: %v\r", float64(index)*100/float64(nn.Config.NumEpochs), "%", acc, "%", index, nn.Config.NumEpochs)

		}
	}()
	tick := time.NewTicker(time.Second * 3)
	go func() {
		for {
			select {
			case <-tick.C:
				nn.SaveWeights()
			}
		}
	}()
	//log.Println("x:", mat.Formatted(x, mat.Prefix(""), mat.Squeeze()))

	// Loop over the number of epochs utilizing
	// backpropagation to train our model.
	for i := 0; i <= nn.Config.NumEpochs; i++ {
		accuracy := nn.CalculateAccuracy(inputTest, labelTest)
		acc = accuracy
		index = i

		//safety brake: avoids artificial intelligence saturation
		if accuracy >= 1.00 && float64(index)*100/float64(nn.Config.NumEpochs) >= 1.60 {
			break
		}

		// Complete the feed forward process.
		hiddenLayerInput := new(mat.Dense)
		hiddenLayerInput.Mul(inputTrain, wHidden)
		addBHidden := func(_, col int, v float64) float64 { return v + bHidden.At(0, col) }
		hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

		hiddenLayerActivations := new(mat.Dense)
		applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
		hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

		outputLayerInput := new(mat.Dense)
		outputLayerInput.Mul(hiddenLayerActivations, wOut)
		addBOut := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }
		outputLayerInput.Apply(addBOut, outputLayerInput)
		output.Apply(applySigmoid, outputLayerInput)

		// Complete the backpropagation.
		networkError := new(mat.Dense)
		networkError.Sub(labelTrain, output)

		slopeOutputLayer := new(mat.Dense)
		applySigmoidPrime := func(_, _ int, v float64) float64 { return sigmoidPrime(v) }
		slopeOutputLayer.Apply(applySigmoidPrime, output)
		slopeHiddenLayer := new(mat.Dense)
		slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

		dOutput := new(mat.Dense)
		dOutput.MulElem(networkError, slopeOutputLayer)
		errorAtHiddenLayer := new(mat.Dense)
		errorAtHiddenLayer.Mul(dOutput, wOut.T())

		dHiddenLayer := new(mat.Dense)
		dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

		// Adjust the parameters.
		wOutAdj := new(mat.Dense)
		wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
		wOutAdj.Scale(nn.Config.LearningRate, wOutAdj)
		wOut.Add(wOut, wOutAdj)

		bOutAdj, err := sumAlongAxis(0, dOutput)
		if err != nil {
			return err
		}
		bOutAdj.Scale(nn.Config.LearningRate, bOutAdj)
		bOut.Add(bOut, bOutAdj)

		wHiddenAdj := new(mat.Dense)
		wHiddenAdj.Mul(inputTrain.T(), dHiddenLayer)
		wHiddenAdj.Scale(nn.Config.LearningRate, wHiddenAdj)
		wHidden.Add(wHidden, wHiddenAdj)

		bHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
		if err != nil {
			return err
		}
		bHiddenAdj.Scale(nn.Config.LearningRate, bHiddenAdj)
		bHidden.Add(bHidden, bHiddenAdj)

	}

	return nil
}

// sumAlongAxis sums a matrix along a particular dimension,
// preserving the other dimension.
func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {

	numRows, numCols := m.Dims()

	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}

	return output, nil
}

// Predict makes a prediction based on a trained
// neural network.
func (nn *NeuralNet) Predict(x *mat.Dense) (*mat.Dense, error) {

	// Check to make sure that our neuralNet value
	// represents a trained model.
	if nn.WHidden == nil || nn.WOut == nil {
		return nil, errors.New("the supplied weights are empty")
	}
	if nn.BHidden == nil || nn.BOut == nil {
		return nil, errors.New("the supplied biases are empty")
	}

	// Define the output of the neural network.
	output := new(mat.Dense)

	// Complete the feed forward process.
	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, nn.WHidden)
	addBHidden := func(_, col int, v float64) float64 { return v + nn.BHidden.At(0, col) }
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	hiddenLayerActivations := new(mat.Dense)
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, nn.WOut)
	addBOut := func(_, col int, v float64) float64 { return v + nn.BOut.At(0, col) }
	outputLayerInput.Apply(addBOut, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)

	return output, nil
}

func (nn *NeuralNet) CalculateAccuracy(inputs *mat.Dense, labels *mat.Dense) float64 {
	var truePosNeg int
	predictions, err := nn.Predict(inputs)
	if err != nil {
		log.Fatal(err)
	}

	numRounds, numOutputs := predictions.Dims()
	for i := 0; i < numRounds; i++ {
		answers := mat.Row(nil, i, labels)
		output := mat.Row(nil, i, predictions)

		var correct int
		for idx, answer := range answers {
			if answer >= 1 {
				answer = 1
			} else if answer < 0 {
				answer = 0
			}
			if output[idx] >= 1 {
				output[idx] = 1
			} else if output[idx] < 0 {
				output[idx] = 0
			}

			if answer == output[idx] {
				correct += 1
			} else {
				correct -= 1
			}
		}
		if correct == numOutputs {
			truePosNeg++
		}
		//api.GetInterface().SetScore(correct)
		//log.Printf(terminal.LogColorHeader+"[predict]:"+terminal.LogColorBlue+"resp:(%v) out:[%v]\n", answers, output)

		//log.Printf(
		//	terminal.LogColorHeader+"[test]:"+terminal.LogColorWarning+"acerts(%v) ouputsLen:%v\n", correct, numOutputs)
		//terminal.NewLog("test", "acerts(%v) ouputsLen:%v", terminal.LevelLog, 6, 6)
		//	terminal.CloneLog("[test]: acerts(%v) ouputsLen:%v\n", correct, numOutputs)
	}

	accuracy := float64(truePosNeg) / float64(numRounds)
	return accuracy

}
