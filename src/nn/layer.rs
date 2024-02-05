use crate::nn::activation::Activation;

pub(crate) trait Layer {
    fn forward(&self, inputs: &Vec<f64>) -> Vec<f64>;
    fn backward(&self, loss_gradient: &Vec<f64>, learning_rate: f64)  -> (Vec<f64>, Vec<f64>);
    fn update(&mut self, weights_gradient: Vec<f64>, gradient: Vec<f64>, learning_rate: f64);
}

struct LayerContent {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
}

pub struct DenseLayer {
    content: LayerContent,
    units: usize,
    activation: Box<dyn Activation>,
}

impl DenseLayer {
    pub fn new(units: usize, activation: Box<dyn Activation>) -> DenseLayer {
        DenseLayer {
            content: LayerContent {
                weights: Vec::new(),
                biases: Vec::new(),
            },
            units,
            activation,
        }
    }
}

impl Layer for DenseLayer {
    fn forward(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let weighted_sum = self.content.weights.iter().map(|weights| {
            weights.iter().zip(inputs.iter()).map(|(w, i)| w * i).sum::<f64>()
        }).collect::<Vec<f64>>();

        let activated = weighted_sum.iter().map(|sum| self.activation.activate(*sum)).collect::<Vec<f64>>();

        activated
    }

    fn backward(&self, loss_gradient: &Vec<f64>, learning_rate: f64) -> (Vec<f64>, Vec<f64>) {
        let bias_gradient = loss_gradient.iter().map(|g| g * self.activation.derivative(*g)).collect::<Vec<f64>>();
        let weights_gradient = bias_gradient.iter().map(|g| g * learning_rate).collect::<Vec<f64>>();

        (bias_gradient, weights_gradient)
    }

    fn update(&mut self, weights_gradient: Vec<f64>, bias_gradient: Vec<f64>, learning_rate: f64) {
        let mut new_weights = Vec::new(); // TODO
        for (i, weights) in self.content.weights.iter().enumerate() {
            new_weights.push(weights.iter().zip(weights_gradient.iter()).map(|(w, g)| w - g * learning_rate).collect::<Vec<f64>>());
        }
        let new_biases = self.content.biases.iter().zip(bias_gradient.iter()).map(|(b, g)| b - g * learning_rate).collect::<Vec<f64>>();

        self.content.weights = new_weights;
        self.content.biases = new_biases;
    }
}

