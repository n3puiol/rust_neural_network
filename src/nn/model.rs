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
            println!("Epoch {}", epoch);
            let mut loss = 0.0;
            for (inputs, targets) in data_set.inputs.iter().zip(data_set.targets.iter()) {
                let mut outputs = inputs.clone();
                for layer in self.layers.iter_mut() {
                    outputs = layer.forward(&outputs);
                }
                println!("outputs: {:?}", outputs);
                loss += self.config.loss.loss(&outputs, targets);
                let loss_gradient = self.config.loss.gradient(&outputs, targets);
                for layer in self.layers.iter_mut() {
                    layer.backward(&loss_gradient, learning_rate);
                }
            }
            println!("Epoch {}: loss {}", epoch, loss);
        }
    }

    fn predict(&mut self, input: &Vector) -> Vector {
        let mut outputs = input.clone();
        for layer in self.layers.iter_mut() {
            outputs = layer.forward(&outputs);
        }
        outputs
    }

    fn evaluate(&self) {
        todo!()
    }
}