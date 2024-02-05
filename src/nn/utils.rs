use crate::mnist_dataset::MnistRecord;

pub struct DataSet {
    pub inputs: Vec<Vec<f64>>,
    pub targets: Vec<Vec<f64>>,
}

pub fn normalize(data: &Vec<MnistRecord>) -> DataSet {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for record in data {
        inputs.push(record.pixels.clone());
        let mut target = vec![0.0; 10];
        target[record.label as usize] = 1.0;
        targets.push(target);
    }

    DataSet {
        inputs,
        targets,
    }
}