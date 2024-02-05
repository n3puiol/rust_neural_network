
pub(crate) trait Activation {
    fn activate(&self, input: f64) -> f64;
    fn derivative(&self, input: f64) -> f64;
}

pub struct Sigmoid;

impl Activation for Sigmoid {
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

pub struct Softmax;

impl Activation for Softmax {
    fn activate(&self, input: f64) -> f64 {
        input.exp()
    }

    fn derivative(&self, _input: f64) -> f64 {
        panic!("Softmax derivative is not implemented");
    }
}