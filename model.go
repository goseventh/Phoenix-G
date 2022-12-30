package neural

import (
	"encoding/json"
	"log"
	"os"

	"gonum.org/v1/gonum/mat"
	"gorm.io/gorm"
)

// save dataset in database
type Dataset struct {
	gorm.Model
	Numbers []uint8
	Result  uint8
}

// NeuralNet contains all of the information
// that defines a trained neural network.
type NeuralNet struct {
	Config  NeuralNetConfig
	WHidden *mat.Dense
	BHidden *mat.Dense
	WOut    *mat.Dense
	BOut    *mat.Dense
}

// NeuralNetConfig defines our neural network
// architecture and learning parameters.
type NeuralNetConfig struct {
	InputNeurons  int
	OutputNeurons int
	HiddenNeurons int
	NumEpochs     int
	LearningRate  float64
}

// NewNetwork initializes a new neural network.
func NewNetwork(config NeuralNetConfig) *NeuralNet {
	return &NeuralNet{Config: config}
}

func (network *NeuralNet) SaveModel() {
	file, err := os.Create("model.json")
	if err != nil {
		panic(err)
	}
	defer file.Close()
	data, err := json.Marshal(network)
	if err != nil {
		panic(err)
	}

	/*	data, err := network.BOut.MarshalBinary()
		if err != nil {
			panic(err)
		}
		log.Println("[save-model]: ", data)*/

	file.Write(data)
	network.SaveWeights()
}

func (network *NeuralNet) SaveWeights() {
	dataBOUT, err := network.BOut.MarshalBinary()
	if err != nil {
		panic(err)
	}
	dataBHidden, err := network.BHidden.MarshalBinary()
	if err != nil {
		panic(err)
	}
	dataWHidden, err := network.WHidden.MarshalBinary()
	if err != nil {
		panic(err)
	}
	dataWOUT, err := network.WOut.MarshalBinary()
	if err != nil {
		panic(err)
	}

	file, err := os.Create("bout.bin")
	if err != nil {
		panic(err)
	}
	file.Write(dataBOUT)
	file.Close()

	file, err = os.Create("bhidden.bin")
	if err != nil {
		panic(err)
	}
	file.Write(dataBHidden)
	file.Close()

	file, err = os.Create("whidden.bin")
	if err != nil {
		panic(err)
	}
	file.Write(dataWHidden)
	file.Close()

	file, err = os.Create("wout.bin")
	if err != nil {
		panic(err)
	}
	file.Write(dataWOUT)
	file.Close()

//	log.Println("[save-model]: ", dataBOUT)

}

func (network *NeuralNet) LoadModel() {
	WHidden := &mat.Dense{}
	BHidden := &mat.Dense{}
	WOut := &mat.Dense{}
	BOut := &mat.Dense{}

	modelBin, err := os.ReadFile("model.json")
	if err != nil {
		panic(err)
	}
	bhiddenBin, err := os.ReadFile("bhidden.bin")
	if err != nil {
		panic(err)
	}

	whiddenBin, err := os.ReadFile("whidden.bin")
	if err != nil {
		panic(err)
	}

	woutBin, err := os.ReadFile("wout.bin")
	if err != nil {
		panic(err)
	}

	boutBin, err := os.ReadFile("bout.bin")
	if err != nil {
		panic(err)
	}

	err = BOut.UnmarshalBinary(boutBin)
	if err != nil {
		panic(err)
	}
	err = WOut.UnmarshalBinary(woutBin)

	if err != nil {
		panic(err)
	}
	err = WHidden.UnmarshalBinary(whiddenBin)

	if err != nil {
		panic(err)
	}
	err = BHidden.UnmarshalBinary(bhiddenBin)

	if err != nil {
		panic(err)
	}

	loadConfig := NeuralNet{}
	err = json.Unmarshal(modelBin, &loadConfig)
	if err != nil {
		panic(err)
	}

	network.BHidden = BHidden
	network.BOut = BOut
	network.WHidden = WHidden
	network.WOut = WOut
	network.Config = loadConfig.Config
	//log.Println("[save-load-model]: ", bOut, modelBin)

	//log.Println("[load-model]: ", network.BOut, modelBin)

	log.Println("[load-model] config:", network.Config)
}
