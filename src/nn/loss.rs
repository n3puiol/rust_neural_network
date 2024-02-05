pub(crate) trait Loss {
    fn loss(&self, outputs: &Vec<f64>, targets: &Vec<f64>) -> f64;

    fn gradient(&self, outputs: &Vec<f64>, targets: &Vec<f64>) -> Vec<f64> {
        let mut gradient = Vec::new();
        for (output, target) in outputs.iter().zip(targets.iter()) {
            gradient.push(output - target);
        }
        gradient
    }
}

pub struct SparseCategoricalCrossentropyLoss;

impl Loss for SparseCategoricalCrossentropyLoss {
    fn loss(&self, outputs: &Vec<f64>, targets: &Vec<f64>) -> f64 {
        let mut loss = 0.0;
        for (output, target) in outputs.iter().zip(targets.iter()) {
            loss += target * output.log(0.0);
        }
        -loss
    }
}