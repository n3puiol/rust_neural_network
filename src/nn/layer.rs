use crate::nn::activation::{Activation};
use crate::nn::loss::Loss;
use crate::nn::neuron::Neuron;
use crate::nn::optimizer::Optimizer;
use crate::nn::utils::{Vector};

pub(crate) trait Layer {
    fn forward(&self, inputs: &Vector) -> Vector;
    fn backward(&mut self, current_output: &Vector, previous_outputs: &Vector, targets: &Vector, optimizer: &Box<dyn Optimizer>, loss: &Box<dyn Loss>, learning_rate: &f64);
}

pub struct InputLayer;

impl InputLayer {
    pub fn new() -> InputLayer {
        InputLayer
    }
}

impl Layer for InputLayer {
    fn forward(&self, inputs: &Vector) -> Vector {
        inputs.clone()
    }

    fn backward(&mut self, _current_output: &Vector, _previous_outputs: &Vector, _targets: &Vector, _optimizer: &Box<dyn Optimizer>, _loss: &Box<dyn Loss>, _learning_rate: &f64) {}
}

pub struct FullyConnectedLayer {
    neurons: Vec<Neuron>,
    activation: Box<dyn Activation>,
}

impl FullyConnectedLayer {
    pub fn new(input_size: usize, units: usize, activation: Box<dyn Activation>) -> FullyConnectedLayer {
        let neurons: Vec<Neuron> = (0..units).map(|_| Neuron::new(input_size)).collect();
        FullyConnectedLayer {
            neurons,
            activation,
        }
    }
}

impl Layer for FullyConnectedLayer {
    fn forward(&self, inputs: &Vector) -> Vector {
        let mut outputs = Vec::new();
        for neuron in self.neurons.iter() {
            outputs.push(self.activation.activate(neuron.forward(inputs)));
        }
        Vector::from(outputs)
    }

    fn backward(&mut self, current_output: &Vector, previous_outputs: &Vector, targets: &Vector, optimizer: &Box<dyn Optimizer>, loss: &Box<dyn Loss>, learning_rate: &f64) {
        for (i, neuron) in self.neurons.iter_mut().enumerate() {
            let neuron_output = current_output[i];
            let neuron_target = targets[i];
            neuron.backward(&neuron_output, previous_outputs, &neuron_target, optimizer, &self.activation, loss, learning_rate);
        }
    }
}

