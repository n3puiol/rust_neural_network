use rand::Rng;
use crate::nn::activation::Activation;
use crate::nn::loss::Loss;
use crate::nn::optimizer::{Optimizer};
use crate::nn::utils::Vector;

pub struct Neuron {
    weights: Vector,
    bias: f64,
}

impl Clone for Neuron {
    fn clone(&self) -> Neuron {
        Neuron {
            weights: self.weights.clone(),
            bias: self.bias,
        }
    }
}

impl Neuron {
    pub(crate) fn new(input_size: usize) -> Neuron {
        let mut weights = Vec::new();
        let mut rng = rand::thread_rng();
        for _ in 0..input_size {
            weights.push(rng.gen_range(-1.0..1.0));
        }
        Neuron {
            weights: Vector::from(weights),
            bias: rand::random(),
        }
    }

    pub(crate) fn forward(&self, inputs: &Vector) -> f64 {
        let mut weighted_sum = 0.0;
        if inputs.len() != self.weights.len() {
            panic!("Input size does not match weight size");
        }
        for (input, weight) in inputs.iter().zip(self.weights.iter()) {
            weighted_sum += input * weight;
        }
        weighted_sum += self.bias;
        weighted_sum
    }

    pub(crate) fn backward(&mut self, neuron_output: &f64, previous_outputs: &Vector, neuron_target: &f64, _optimizer: &Box<dyn Optimizer>, activation: &Box<dyn Activation>, _loss: &Box<dyn Loss>, learning_rate: &f64) {
        let delta_output = neuron_output - neuron_target;
        for (i, weight) in self.weights.iter_mut().enumerate() {
            let gradient = 2.0 * delta_output * activation.derivative(*neuron_output) * previous_outputs[i];
            *weight -= learning_rate * gradient;
        }
    }
}