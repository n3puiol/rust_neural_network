use std::any::Any;

pub(crate) trait Activation {
    fn as_mut_any(&mut self) -> &mut dyn Any;
    fn activate(&self, input: f64) -> f64;
    fn derivative(&self, input: f64) -> f64;
}

pub struct Sigmoid;

impl Activation for Sigmoid {
    fn as_mut_any(&mut self) -> &mut dyn Any {
        self
    }

    fn activate(&self, input: f64) -> f64 {
        1.0 / (1.0 + (-input).exp())
    }

    fn derivative(&self, input: f64) -> f64 {
        let sigmoid = self.activate(input);
        sigmoid * (1.0 - sigmoid)
    }
}

pub struct ReLU;

impl Activation for ReLU {
    fn as_mut_any(&mut self) -> &mut dyn Any {
        self
    }

    fn activate(&self, input: f64) -> f64 {
        if input > 0.0 {
            input
        } else {
            0.0
        }
    }

    fn derivative(&self, input: f64) -> f64 {
        if input > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

// pub struct Softmax {
//     z_sum: f64,
// }
//
// impl Softmax {
//     pub(crate) fn new() -> Softmax {
//         Softmax {
//             z_sum: 0.0,
//         }
//     }
//
//     pub fn compute_z_sum(&mut self, inputs: &Vector) {
//         let exp_values: Vec<f64> = inputs.iter().map(|&x| E.powf(x)).collect();
//         self.z_sum = exp_values.iter().sum();
//     }
// }
//
// impl Activation for Softmax {
//     fn as_mut_any(&mut self) -> &mut dyn Any {
//         self
//     }
//
//     fn activate(&self, input: f64) -> f64 {
//         E.powf(input) / self.z_sum
//     }
//
//     fn derivative(&self, input: f64) -> f64 {
//         let softmax_output = self.activate(input);
//         softmax_output * (1.0 - softmax_output)
//     }
// }