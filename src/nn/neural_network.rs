use crate::nn::layer::Layer;

pub struct NeuralNetwork {
    layers: Vec<Box<dyn Layer>>,
}

// impl NeuralNetwork {
//     pub(crate) fn new(layer_sizes: &[usize]) -> Self {
//         // Initialize the neural network layers with random weights and biases
//         // ...
//
//         NeuralNetwork { layers }
//     }
//
//     fn forward(&self, input: Vec<f64>) -> Vec<f64> {
//         // Implement the forward pass
//         // ...
//
//         // Return the output of the neural network
//         // ...
//     }
//
//     fn train(&mut self, input: Vec<f64>, target: Vec<f64>, learning_rate: f64) {
//         // Implement the training/backpropagation algorithm
//         // ...
//
//         // Update weights and biases
//         // ...
//     }
// }
