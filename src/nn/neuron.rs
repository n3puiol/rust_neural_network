use crate::nn::utils::Vector;

pub struct Neuron {
    weights: Vector,
    bias: f64,
    weighted_sum: f64,
}

impl Clone for Neuron {
    fn clone(&self) -> Neuron {
        Neuron {
            weights: self.weights.clone(),
            bias: self.bias,
            weighted_sum: self.weighted_sum,
        }
    }
}

impl Neuron {
    pub(crate) fn new(input_size: usize) -> Neuron {
        let mut weights = Vec::new();
        for _ in 0..input_size {
            weights.push(rand::random());
        }
        Neuron {
            weights: Vector::from(weights),
            bias: rand::random(),
            weighted_sum: 0.0,
        }
    }

    pub(crate) fn forward(&mut self, inputs: &Vector) -> f64 {
        self.weighted_sum = 0.0;
        for (weight, input) in self.weights.iter().zip(inputs.iter()) {
            self.weighted_sum += weight * input;
        }
        self.weighted_sum += self.bias;
        self.weighted_sum
    }

    pub(crate) fn backward(&mut self, loss: &f64, learning_rate: f64) {
        for weight in self.weights.iter_mut() {
            *weight -= loss * learning_rate;
        }
        self.bias -= loss * learning_rate;
    }
}