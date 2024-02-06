use crate::nn::activation::Activation;
use crate::nn::neuron::Neuron;
use crate::nn::utils::{Vector};

pub(crate) trait Layer {
    fn forward(&mut self, inputs: &Vector) -> Vector;
    fn backward(&mut self, loss: &Vector, learning_rate: f64);
}

pub struct FullyConnectedLayer {
    neurons: Vec<Neuron>,
    activation: Box<dyn Activation>,
}

impl FullyConnectedLayer {
    pub fn new(input_size: usize, units: usize, activation: Box<dyn Activation>) -> FullyConnectedLayer {
        FullyConnectedLayer {
            neurons: vec![Neuron::new(input_size); units],
            activation,
        }
    }
}

impl Layer for FullyConnectedLayer {
    fn forward(&mut self, inputs: &Vector) -> Vector {
        let mut outputs = Vec::new();
        for neuron in self.neurons.iter_mut() {
            outputs.push(self.activation.activate(neuron.forward(inputs)));
        }
        Vector::from(outputs)
    }

    fn backward(&mut self, loss: &Vector, learning_rate: f64) {
        for (neuron, loss) in self.neurons.iter_mut().zip(loss.iter()) {
            neuron.backward(loss, learning_rate);
        }
    }
}

