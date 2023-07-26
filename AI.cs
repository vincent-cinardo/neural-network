namespace Cintech.AI
{
    public class Synapse
    {
        static Random random = new Random();
        public Neuron forwardNeuron;
        public Neuron backwardNeuron;
        public double weight;
        public double input;
        public double weightGradient, biasGradient;
        public Synapse(Neuron backwardNeuron, Neuron forwardNeuron)
        {
            this.backwardNeuron = backwardNeuron;
            this.forwardNeuron = forwardNeuron;
            this.weight = random.NextDouble() * 2 - 1;
        }
    }
    public class Neuron
    {
        public List<Synapse> forwardSynapses;
        public List<Synapse> backwardSynapses;
        public double bias;
        public double epochs;
        private double learningRate;

        public Neuron(double epochs, double learningRate)
        {
            forwardSynapses = new List<Synapse>();
            backwardSynapses = new List<Synapse>();

            this.epochs = epochs;
            this.learningRate = learningRate;

            Random random = new Random();
            bias = random.NextDouble() * 2 - 1;
        }

        //By default, linear activation function
        public virtual double Activate(double x)
        {
            return x;
        }

        public virtual double Derive(double x)
        {
            return 1;
        }

        public int ForwardCount()
        {
            return forwardSynapses.Count;
        }

        public int BackwardCount()
        {
            return backwardSynapses.Count;
        }

        public void FeedForward()
        {
            for (int i = 0; i < forwardSynapses.Count; i++)
            {
                forwardSynapses[i].input = Predict();
            }
        }
        public static double Tanh(double sum)
        {
            return Math.Tanh(sum);
        }

        public static double DeriveTanh(double sum)
        {
            return 1 - (sum * sum);
        }

        public void Connect(Neuron neuron)
        {
            Synapse synapse = new Synapse(this, neuron);
            forwardSynapses.Add(synapse);
            neuron.backwardSynapses.Add(synapse);
        }

        public void FeedInput(double input)
        {
            forwardSynapses[0].input = input;
        }

        public void Train()
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                for (int i = 0; i < BackwardCount(); i++)
                {
                    backwardSynapses[i].weight = backwardSynapses[i].weight - learningRate * backwardSynapses[i].weightGradient;
                }
            }
        }

        public double WeightedSum()
        {
            double sum = 0;
            for (int i = 0; i < BackwardCount(); i++)
            {
                sum += backwardSynapses[i].input * backwardSynapses[i].weight;
            }
            sum += bias;
            return sum;
        }

        public double Predict()
        {
            return Activate(WeightedSum());
        }
    }

    class TanhNeuron : Neuron
    {
        public TanhNeuron(double epochs, double learningRate) : base(epochs, learningRate) { }

        public override double Activate(double x)
        {
            return Tanh(x);
        }

        public override double Derive(double x)
        {
            return DeriveTanh(x);
        }
    }
    public class Layer
    {
        public List<Neuron> neurons;
        protected int numberNeuron, epochs;
        protected double learningRate;
        
        public Layer(int numberNeuron, int epochs, double learningRate)
        {
            neurons = new List<Neuron>();
            this.epochs = epochs;
            this.learningRate = learningRate;
            this.numberNeuron = numberNeuron;
            Initialize();
        }

        public virtual void Initialize()
        {
            for (int i = 0; i < numberNeuron; i++)
            {
                neurons.Add(new Neuron(epochs, learningRate));
            }
        }

        public void ConnectBackward(Layer backwardLayer)
        {
            for (int i = 0; i < backwardLayer.NeuronCount(); i++)
            {
                for (int j = 0; j < NeuronCount(); j++)
                {
                    backwardLayer[i].Connect(this[j]);
                }
            }
        }

        public Neuron this[int i]
        {
            set { neurons[i] = value; }
            get { return neurons[i]; }
        }

        public int NeuronCount()
        {
            return neurons.Count;
        }

        public void FeedForward()
        {
            for (int i = 0; i < NeuronCount(); i++)
            {
                for (int j = 0; j < this[i].ForwardCount(); j++)
                {
                    this[i].FeedForward();
                }
            }
        }

        public double[] Predict()
        {
            double[] activationArray = new double[NeuronCount()];

            for (int i = 0; i < activationArray.Length; i++)
            {
                activationArray[i] = this[i].Predict();
            }

            return activationArray;
        }
    }

    public class TahnLayer : Layer
    {
        public TahnLayer(int numberNeuron, int epochs, double learningRate) : base(numberNeuron, epochs, learningRate) { }

        public override void Initialize()
        {
            for (int i = 0; i < numberNeuron; i++)
            {
                neurons.Add(new TanhNeuron(epochs, learningRate));
            }
        }
    }

    public class NeuralNetwork
    {
        public int numInputs;
        public int numOutputs;
        public List<Layer> layers;
        public NeuralNetwork(int numInputs, int numOutputs, int hiddenLayers, int width, int epochs, double learningRate)
        {
            this.numInputs = numInputs;
            this.numOutputs = numOutputs;
            layers = new List<Layer>();

            layers.Add(new Layer(numInputs, epochs, learningRate));

            for (int i = 1; i < hiddenLayers; i++)
            {
                layers.Add(new TahnLayer(width, epochs, learningRate));
                layers[i].ConnectBackward(layers[i - 1]);
            }

            layers.Add(new Layer(numOutputs, epochs, learningRate));
            layers[layers.Count - 1].ConnectBackward(layers[layers.Count - 2]);
        }

        public void Train(double[] inputs, double[] outputs)
        {
            if (inputs.Length != numInputs)
            {
                Console.Write("Array's length must match the number of neurons in the input layer.");
                return;
            }
            
            for (int i = 0; i < inputs.Length; i++)
            {
                layers[0][i].FeedInput(inputs[i]);
            }

            for (int i = 1; i < layers.Count - 1; i++)
            {
                layers[i].FeedForward();
            }

            Backpropagate(outputs);
        }

        public double[] Predict(double[] inputs)
        {
            if (inputs.Length != numInputs)
            {
                Console.Write("Array's length must match the number of neurons in the input layer.");
                return Array.Empty<double>();
            }

            for (int i = 0; i < inputs.Length; i++)
            {
                layers[0][i].FeedInput(inputs[i]);
            }

            for (int i = 1; i < layers.Count - 1; i++)
            {
                layers[i].FeedForward();
            }

            double[] output = layers[layers.Count - 1].Predict();
            return output;
        }

        private void Backpropagate(double[] rawOutput)
        {
            double error = 0.0;
            double finalError = 0.0;
            int last = layers.Count - 1;
            double weightGradient, sumWeightGradients, prediction, desired;

            // Calculate the error and update gradients for the output layer
            for (int j = 0; j < layers[last].NeuronCount(); j++)
            {
                desired = rawOutput[j];
                prediction = layers[last][j].Predict();
                weightGradient = 2 * (prediction - desired);

                for (int k = 0; k < layers[last][j].backwardSynapses.Count; k++)
                {
                    layers[last][j].backwardSynapses[k].weightGradient = weightGradient * layers[last][j].backwardSynapses[k].weight;
                }
            }

            // Propagate the error back through hidden layers (chain rule)
            for (int i = last - 1; i > 0; i--)
            {
                for (int j = 0; j < layers[i].NeuronCount(); j++)
                {
                    sumWeightGradients = 0.0;
                    for (int k = 0; k < layers[i][j].forwardSynapses.Count; k++)
                    {
                        sumWeightGradients += layers[i][j].forwardSynapses[k].weightGradient;
                    }

                    weightGradient = Neuron.DeriveTanh(layers[i][j].WeightedSum()) * sumWeightGradients ;

                    for (int k = 0; k < layers[i][j].backwardSynapses.Count; k++)
                    {
                        if (i == 1)
                        {
                            layers[i][j].backwardSynapses[k].weightGradient = weightGradient * layers[i][j].backwardSynapses[k].input;
                        }
                        else layers[i][j].backwardSynapses[k].weightGradient = weightGradient * layers[i][j].backwardSynapses[k].weight;
                    }
                }
            }

            // Train using all errors found;
            for (int i = last; i > 0; i--)
            {
                for (int j = 0; j < layers[i].NeuronCount(); j++)
                {
                    layers[i][j].Train();
                }
            }

            for (int i = 0; i < layers[last].NeuronCount(); i++)
            { 
                error = layers[last][i].Predict() - rawOutput[i];
                error *= error;
                finalError += error;
            }

            double[] output = layers[last].Predict();

            Console.Write("Prediciton is now [");
            for (int i = 0; i < layers[last].NeuronCount(); i++)
            {
                Console.Write(layers[last][i].Predict() + ", ");
            }
            Console.WriteLine("\b\b]");

            Console.Write("Expected was ");
            for (int i = 0; i < output.Length; i++)
            {
                Console.Write(output[i] + ", ");
            }
            Console.WriteLine("\b\b]");
            Console.WriteLine("Error is now " + finalError + "\n");
        }
    }
}