use crate::nn::utils::Vector;

pub(crate) trait Loss {
    fn single_loss(&self, output: f64, target: f64) -> f64;
    fn total_loss(&self, outputs: &Vector, targets: &Vector) -> f64;
    fn gradient(&self, outputs: &Vector, targets: &Vector) -> Vector;
}

pub struct MeanSquaredErrorLoss;

impl Loss for MeanSquaredErrorLoss {
    fn single_loss(&self, output: f64, target: f64) -> f64 {
        (output - target).powi(2)
    }

    fn total_loss(&self, outputs: &Vector, targets: &Vector) -> f64 {
        let mut loss = 0.0;
        for (output, target) in outputs.iter().zip(targets.iter()) {
            loss += self.single_loss(*output, *target);
        }
        loss
    }

    fn gradient(&self, outputs: &Vector, targets: &Vector) -> Vector {
        let mut gradient = Vec::new();
        for (output, target) in outputs.iter().zip(targets.iter()) {
            gradient.push(2.0 * (output - target));
        }
        Vector::from(gradient)
    }
}

// pub struct SparseCategoricalCrossentropyLoss;
//
// impl Loss for SparseCategoricalCrossentropyLoss {
//     fn total_loss(&self, outputs: &Vector, targets: &Vector) -> f64 {
//         let mut loss = 0.0;
//         for (output, target) in outputs.iter().zip(targets.iter()) {
//             loss += -target * output.ln();
//         }
//         loss
//     }
//
//     fn gradient(&self, outputs: &Vector, targets: &Vector) -> Vector {
//         let mut gradient = Vec::new();
//         for (output, target) in outputs.iter().zip(targets.iter()) {
//             gradient.push(-target / output);
//         }
//         Vector::from(gradient)
//     }
// }