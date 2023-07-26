# Neural Network

Sidetracked from my game, was interested in finding out how a neural network works. I was able to create a network that uses the tanh activation functions for hidden layers and a linear activation function for the output layer. The network takes in an array of doubles for the inputs and an array of doubles for the expected output. Other hyperparameters can be specified as well.

## How to use:

```
int numInputs = 4;
int numOutputs = 1;
int hiddenLayers = 5;
int width = 5;
int epochs = 100;
double learningRate = 0.0001;
NeuralNetwork net = new NeuralNetwork(numInputs, numOutputs, hiddenLayers, width, epochs, 0.0001);

// Create arrays of required lengths, numInputs and numOutputs.
double[] inputs = { 0.05, 0.03, -0.05, 0.01 };
double[] outputs = { 0.06 };

// Console outputs show convergance towards the expected output(s) array here.
while (true)
{
    net.Train( inputs, outputs);
    Thread.Sleep(10);
    Console.Clear();
}
```
