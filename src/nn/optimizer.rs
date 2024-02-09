use crate::nn::activation::Activation;
use crate::nn::utils::Vector;

pub(crate) trait Optimizer {
    fn optimize(&self, input: &Vector, output: &Vector, targets: &Vector, weighted_sum: f64, desired_output: f64, weights: &mut Vector, bias: &mut f64, loss: &f64, activation: &Box<dyn Activation>, learning_rate: f64);
}

pub struct GradientDescentOptimizer;

impl Optimizer for GradientDescentOptimizer {
    fn optimize(&self, previous_layer_inputs: &Vector, _outputs: &Vector, _targets: &Vector, weighted_sum: f64, desired_output: f64, weights: &mut Vector, bias: &mut f64, _loss: &f64, activation: &Box<dyn Activation>, learning_rate: f64) {
        //     loop over the inputs of previous layer
        //     compute derivative of loss with respect to weights
        //     the derivative is composed of:
        //     - the derivative of the weighted sum with respect to the weights (=previous layer input)
        //     - the derivative of the current layer activation with respect to the weighted sum (=activation derivative of weighted sum)
        //     - the derivative of the loss with respect to the current layer activation (2*(current layer activation-target))
        //     all together this looks like: cost_derivative = previous_layer_input * activation_derivative(weighted_sum) * 2*(current_layer_activation - target)
        //     to actually get the gradient, we need to average the cost/loss over all inputs:
        //     gradient = sum(cost_derivative) / len(inputs)
        //     but to compute the full gradient, we will also need all the other derivatives with respect to all the other weights and biases in the entire network.
        //     so we need the bias derivative as well:
        //     bias_derivative = activation_derivative(weighted_sum) * 2*(current_layer_activation - target)
        for _input in previous_layer_inputs.iter() {
            let gradient = 1.0/ previous_layer_inputs.len() as f64;
            for weight in weights.iter_mut() {
                *weight -= learning_rate * gradient;
            }
            *bias -= learning_rate * activation.derivative(weighted_sum) * 2.0 * (activation.activate(weighted_sum) - desired_output);
        }
    }
}

// pub struct AdamOptimizer;
//
// impl Optimizer for AdamOptimizer {
//     fn optimize(&self) {
//         todo!()
//     }
// }