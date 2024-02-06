use crate::nn::utils::Vector;

pub(crate) trait Loss {
    fn loss(&self, outputs: &Vector, targets: &Vector) -> f64;

    fn gradient(&self, outputs: &Vector, targets: &Vector) -> Vector;
}

pub struct SparseCategoricalCrossentropyLoss;

impl Loss for SparseCategoricalCrossentropyLoss {
    fn loss(&self, outputs: &Vector, targets: &Vector) -> f64 {
        let mut loss = 0.0;
        for (output, target) in outputs.iter().zip(targets.iter()) {
            loss += target * output.log(0.0);
        }
        -loss
    }

    fn gradient(&self, outputs: &Vector, targets: &Vector) -> Vector {
        let mut gradient = Vec::new();
        for (output, target) in outputs.iter().zip(targets.iter()) {
            gradient.push(-target / output);
        }
        Vector::from(gradient)
    }
}