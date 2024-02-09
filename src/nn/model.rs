use crate::nn::layer::Layer;
use crate::nn::loss::Loss;
use crate::nn::optimizer::Optimizer;
use crate::nn::utils::{DataSet, Vector};

pub struct ModelConfig {
    pub optimizer: Box<dyn Optimizer>,
    pub loss: Box<dyn Loss>,
}

pub(crate) trait Model {
    fn new(model_config: ModelConfig) -> Self;
    fn summary(&self);
    fn compile(&self);
    fn fit(&mut self, data_set: &DataSet, epochs: usize, learning_rate: f64);
    fn predict(&mut self, input: &Vector) -> Vector;
    fn evaluate(&self);
}

pub struct SequentialModel {
    config: ModelConfig,
    layers: Vec<Box<dyn Layer>>,
}

impl SequentialModel {
    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }
}

impl Model for SequentialModel {
    fn new(model_config: ModelConfig) -> Self {
        SequentialModel {
            config: model_config,
            layers: Vec::new(),
        }
    }
    fn summary(&self) {
        todo!()
    }

    fn compile(&self) {
        todo!()
    }

    fn fit(&mut self, data_set: &DataSet, epochs: usize, learning_rate: f64) {
        for epoch in 0..epochs {
            let mut loss = 0.0;
            let mut outputs = Vec::new();
            for (inputs, targets) in data_set.inputs.iter().zip(data_set.targets.iter()) {
                outputs.push(inputs.clone());
                for layer in self.layers.iter() {
                    let current_output = outputs.last().unwrap();
                    outputs.push(layer.forward(current_output));
                }
                loss += self.config.loss.total_loss(outputs.last().unwrap(), targets);
                for (i, layer) in self.layers.iter_mut().enumerate().rev() {
                    let current_output = outputs.pop().unwrap();
                    let previous_output = outputs.last().unwrap();
                    let layer_targets = if i == 0 { targets } else { previous_output };
                    layer.backward(&current_output, previous_output, layer_targets, &self.config.optimizer, &self.config.loss, &learning_rate);
                }
                outputs.clear();
            }
            println!("Epoch {}: loss {}", epoch, loss.round());
        }
    }

    fn predict(&mut self, input: &Vector) -> Vector {
        let mut outputs = input.clone();
        for layer in self.layers.iter_mut() {
            outputs = layer.forward(&outputs);
        }
        for output in outputs.iter_mut() {
            *output = (*output * 100.0).round() / 100.0;
        }
        outputs
    }

    fn evaluate(&self) {
        todo!()
    }
}